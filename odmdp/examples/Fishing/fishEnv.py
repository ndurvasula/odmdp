import numpy as np
from scipy.optimize import fsolve

def f(x,k):
    return x**(k+1) - 2*x + 1

def count(hours):
    return np.random.normal(hours*FPH, np.sqrt(hours*FSTD**2))

DAYS = 100
TYPES = 5 #1 indexed
e = fsolve(f, .5, (TYPES))[0]
P = np.array([e**i for i in range(1,TYPES+1)])

#Fish per hour and STD on FPH
FPH = 20
FSTD = 5

#Pricing

#Gas pricing in dollars per gallon
GAS_MEAN = 3
GAS_PERIOD = 50 #in days
GAS_AMPLITUDE = 1

#Number of gallons in tank
GAL = 50

#Fish pricing
BASE = 1 #Expected price for worst fish
STD = .1 #Standard deviation on fish price
MULT = 1.5 #Scale by mult^(type-1)
PERIOD = 100 #Period for type 1 fish
RATE = 1/2 #Ratio of successive type periods

def price(typ, quality, day):
    scale = (typ+quality)*MULT**(typ-1)
    pd = PERIOD*RATE**(typ-1)
    mean = scale*np.sin(pd*day/(2*np.pi))
    return np.random.normal(mean,STD)

def gas(time):
    return GAL*(np.normal(

def transition(hours,time):
    fish = np.random.multinomial(np.round(count(hours)),P)
    return fish
                            

