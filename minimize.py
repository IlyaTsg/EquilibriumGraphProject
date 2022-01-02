from scipy import optimize

def f(x, y, z):
    return x(x+50+10(x+z)) + y(10(x+z)+y+50) + z(10(y+z)+z+10+10(x+z))

print(optimize.minimize(f(), [5]))