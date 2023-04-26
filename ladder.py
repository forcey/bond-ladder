import numpy as np
from scipy.optimize import linprog

class Bond:
    def __init__(self, face_value, price, coupon, term):
        self.face_value = face_value
        self.price = price
        self.coupon = coupon
        self.term = term

    # returns a vector of monthly cash flows.
    # the coupon is paid every 6 months.
    def cash_flow(self):
        array = np.zeros(self.term + 1)
        array[0] = -self.price
        for i in range(1, self.term + 1):
            if i % 6 == 0:
                array[i] = self.coupon / 2
        array[self.term] += self.face_value
        return array

class Mortgage:
    def __init__(self, balance, rate, payment):
        self.balance = balance
        self.rate = rate / 1200 # annual percentage to monthly rate
        self.payment = payment

    # returns a 2d array of monthly cash flows.
    # each row is a scenario, each column is a month.
    def cash_flow(self):
        balances = [self.balance]
        balance = self.balance
        while balance >= 0:
            balance += balance * self.rate - self.payment
            if balance >= 0:
                balances.append(balance)
        term = len(balances)
        array = np.zeros((term, term))
        for i in range(term):
            array[i] = [self.payment] * i + [balances[i]] + [0] * (term - i - 1)
        return array

def pad_array(arr, length):
    return np.pad(arr, (0, length - len(arr)), mode='constant', constant_values=(0,))

def stack_arrays(arrays):
    length = max([len(arr) for arr in arrays])
    return np.vstack([pad_array(arr, length) for arr in arrays])

def calculate_ladder(bonds, mortgage):
    # calculate the cash flow for each bond and mortgage
    bond_cash_flows = stack_arrays([bond.cash_flow() for bond in bonds])
    mortgage_cash_flows = mortgage.cash_flow()

    # calculate the dimensions
    variables = 1 + len(bonds) + len(mortgage_cash_flows)
    term = len(mortgage_cash_flows[0]) # ignore bonds that are longer than the mortgage

    # pad the cash flow with zeros
    cash = np.array([1] + [0] * (term - 1))
    if term > len(bond_cash_flows[0]):
        bond_cash_flows = np.hstack((bond_cash_flows, np.zeros((len(bonds), term - len(bond_cash_flows[0])))))
    else:
        bond_cash_flows = bond_cash_flows[:, :term]

    cash_flow = np.vstack((cash, bond_cash_flows, -mortgage_cash_flows))
    
    # cumulative cash flow, transposed for next step: each column is a variable, each row is a month
    cum_cash_flow = np.cumsum(np.transpose(cash_flow), axis=0)
    # print(cum_cash_flow)

    # solve the linear programming
    A_ub = -cum_cash_flow
    b_ub = np.zeros(term)
    # total of mortgage variables must be 1 (only one scenario to be chosen)
    A_eq = np.array([[0] * (1 + len(bonds)) + [1] * len(mortgage_cash_flows)])
    b_eq = np.array([1])

    # Objective function: minimize the initail cash
    c = np.zeros(variables)
    c[0] = 1

    # All variables must be non-negative integers.
    bounds = [(0, None) for _ in range(variables)]
    integrality=np.ones((variables))

    # Solve the linear programming problem
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', integrality=integrality)
    # print(res)

    print("total cost = ", res.fun)
    print("cash: ", res.x[0])
    print("purchases: ", res.x[1:len(bonds)+1] * 100)
    print("securities cashflow: ", np.transpose(bond_cash_flows) @ res.x[1:1+len(bonds)])
    print("mortgage cashflow: ", np.transpose(mortgage_cash_flows) @ res.x[1+len(bonds):])
    print("balances: ", cum_cash_flow @ res.x)

    return res

if __name__ == '__main__':
    b = [
        Bond(100, 95.419, 0, 1 * 12),
        Bond(100, 99.849, 3.875, 2 * 12),
        Bond(100, 99.831, 3.75, 3 * 12),
        Bond(100, 99.818, 3.625, 5 * 12),
        Bond(100, 99.993, 3.625, 7 * 12),
        Bond(100, 99.058, 3.5, 10 * 12),
        Bond(100, 98.601, 3.875, 20 * 12),
    ]
    m = Mortgage(1000000, 3., 4216)

    np.set_printoptions(precision=2, suppress=True)
    calculate_ladder(b, m)
