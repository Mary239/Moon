# units: kilometer, kilogram, second
# coordinate system: the origin is in the center of the moon
#at the initial moment oX is directed at the Moon, oY is directed at the North pole
import math
import sys
import matplotlib.pyplot as plt
import pylab
from numpy import *
output = open('moontoearth.txt', 'w')
INPUT_FILE = 'from_4_to_5.txt'
gEarth = 0.00981
gMoon = 0.00162
rEarth = 6375
rMoon = 1738
GM = gEarth * rEarth * rEarth#G * Earth_mass
Gm = gMoon * rMoon * rMoon #G * Moon_mass
R = 384405  # radius of the Moon's orbit
pi = math.pi
Tmoon = 2 * pi * math.sqrt(R * R * R / GM)
dryMass = 10300  #dry mass of the accelerating stage
F = 95.75 #jet force of the accelerating stage
u = 3.05  #actual exhaust velocity of the accelerating stage
q = F / u  #fuel consumption (kilograms per second) of the accelerating stage


class Vector:
    def plus(a, b):
        # returns the sum of a and b
        ans = Vector()
        ans.x = a.x + b.x
        ans.y = a.y + b.y
        ans.z = a.z + b.z
        return ans

    def minus(a, b):
        # returns the difference between a and b
        ans = Vector()
        ans.x = a.x - b.x
        ans.y = a.y - b.y
        ans.z = a.z - b.z
        return ans

    def absV(a):
        # returns the absolute value of a
        return math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)

    def mult(k, a):
        # returns product of scalar k and vector a
        ans = Vector()
        ans.x = k * a.x
        ans.y = k * a.y
        ans.z = k * a.z
        return ans

    def angle(v, u):
        # returns value of the angle between v and u
        a = Vector.absV(v)
        b = Vector.absV(u)
        c = v.x * u.x + v.y * u.y + v.z * u.z
        return math.acos(c / a / b)

    def copy(a):
        ans = Vector()
        ans.x = a.x
        ans.y = a.y
        ans.z = a.z
        return ans


class RVTME:
    # contains current position, velocity, time total mass and boolean state of the engine (0 - off, q - acceleration)
    # (0 - off, q - acceleration)
    def copy(rvtme):
        ans = RVTME()
        ans.r = Vector.copy(rvtme.r)
        ans.v = Vector.copy(rvtme.v)
        ans.t = rvtme.t
        ans.m = rvtme.m
        ans.engine = rvtme.engine
        return ans


def moonPosition(time):
    # returns the vector of Moon's position
    global R, pi, Tmoon
    ans = Vector()
    ans.x = R * math.cos(2 * pi * time / Tmoon)
    ans.y = R * math.sin(2 * pi * time / Tmoon)
    ans.z = 0
    return ans


def moonV(time):
    # returns the vector of Moon's velocity
    global Tmoon, pi, R
    ans = Vector()
    ans.x = -2 * pi * R / Tmoon * math.sin(2 * pi * time / Tmoon)
    ans.y = 2 * pi * R / Tmoon * math.cos(2 * pi * time / Tmoon)
    ans.z = 0
    return ans


def timestep(a, deltaT=0.00005):
    # returns non-constant timestep so as to make our model more accurate
    return deltaT / Vector.absV(a)


def acc(r, v, time, mass, engine):
    # returns the acceleration of the apparatus
    global GM, Gm, q, F, q2, F2
    aEarth = Vector.mult(-GM / (Vector.absV(r) * Vector.absV(r) * Vector.absV(r)), r)
    moon = Vector.minus(r, moonPosition(time))
    aMoon = Vector.mult(-Gm / (Vector.absV(moon) * Vector.absV(moon) * Vector.absV(moon)), moon)
    aEngine = Vector()
    if engine == 0:
        aEngine.x = 0
        aEngine.y = 0
        aEngine.z = 0
    if engine == q:
        aEngine = Vector.mult(F / mass / Vector.absV(v), v)
        # let jet force and velocity be co-directed
    return Vector.plus(aEngine, Vector.plus(aEarth, aMoon))


def nextRVTME(previous, timestep):
    # returns the next value of position and velocity of the apparatus (by the Runge-Kutta method)
    ans = RVTME()
    v1 = Vector.mult(timestep, acc(previous.r, previous.v, previous.t, previous.m, previous.engine))
    r1 = Vector.mult(timestep, previous.v)
    v2 = Vector.mult(timestep,
                     acc(Vector.plus(previous.r, Vector.mult(0.5, v1)), Vector.plus(previous.v, Vector.mult(0.5, v1)),
                         previous.t + timestep / 2, previous.m - 0.5 * previous.engine * timestep, previous.engine))
    r2 = Vector.mult(timestep, Vector.plus(previous.v, Vector.mult(0.5, v2)))
    v3 = Vector.mult(timestep,
                     acc(Vector.plus(previous.r, Vector.mult(0.5, v2)), Vector.plus(previous.v, Vector.mult(0.5, v2)),
                         previous.t + timestep / 2, previous.m - 0.5 * previous.engine * timestep, previous.engine))
    r3 = Vector.mult(timestep, Vector.plus(previous.v, Vector.mult(0.5, v3)))
    v4 = Vector.mult(timestep, acc(Vector.plus(previous.r, v3), Vector.plus(previous.v, v2),
                                   previous.t + timestep, previous.m - previous.engine * timestep, previous.engine))
    r4 = Vector.mult(timestep, Vector.plus(previous.v, v4))
    ans.r = Vector.plus(previous.r, Vector.mult(1.0 / 6,
                                                Vector.plus(r1, Vector.plus(r2, Vector.plus(r2, Vector.plus(r3,
                                                                                                            Vector.plus(
                                                                                                                r3,
                                                                                                                r4)))))))
    ans.v = Vector.plus(previous.v, Vector.mult(1.0 / 6,
                                                Vector.plus(v1, Vector.plus(v2, Vector.plus(v2, Vector.plus(v3,
                                                                                                            Vector.plus(
                                                                                                                v3,
                                                                                                                v4)))))))
    ans.t = previous.t + timestep
    ans.m = previous.m - timestep * previous.engine
    ans.engine = previous.engine
    return ans;


def test(rvtme):
    # returns the distance to the Earth when our velocity is parallel to the Earth's surface
    angle = pi / 2 - Vector.angle(rvtme.r, rvtme.v)
    while (angle < 0) or (Vector.absV(rvtme.r) > 100000):
        rvtme = nextRVTME(rvtme, timestep(acc(rvtme.r, rvtme.v, rvtme.t, rvtme.m, rvtme.engine)))
        angle = pi / 2 - Vector.angle(rvtme.r, rvtme.v)

    return Vector.absV(rvtme.r)




def main():
    global dryMass, GM, Gm, q, q2, R, rMoon, pi, u
    f = open(INPUT_FILE, 'r').readlines()
    mmm = array([[float(i) for i in f[k].split()] for k in range((len(f)))])
    string = open('from_2_to_3_and_5.txt').readlines()
    mm = array([[float(i) for i in string[k].split()] for k in range((len(string)))])
    mSpent = mm[0][4]  # Fuel in the SM, spent on the flight to the Moon
    v = mmm[0][0]/1000
    h = mmm[0][1]
    print(v,h)
    mFuel = 17700 - mSpent  # Remaining fuel in the SM
    # We calculate the appropriate start point, based on the data of the output file of stage 4
    x = R + (rMoon + h / 1000) * math.cos(math.asin(math.sqrt(GM * (rMoon + h / 1000) / 2 / Gm / R)))
    y = (rMoon + h / 1000) * math.sqrt(GM * (rMoon + h / 1000) / 2 / Gm / R)
    z = 0
    vx = (math.sqrt(GM * (rMoon + h / 1000) / 2 / Gm / R)) * v
    vy = 1.0184 - math.cos(math.asin(math.sqrt(GM * (rMoon + h / 1000) / 2 / Gm / R))) * v
    vz = 0
    rvtme = RVTME()
    rvtme.r = Vector()
    rvtme.v = Vector()
    rvtme.r.x = x
    rvtme.r.y = y
    rvtme.r.z = z
    rvtme.v.x = vx
    rvtme.v.y = vy
    rvtme.v.z = vz
    rvtme.t = 0
    rvtme.m = dryMass  + mFuel

    deltaV = -Vector.absV(Vector.minus(rvtme.v, moonV(rvtme.t))) + \
             math.sqrt(2 * Gm / Vector.absV(Vector.minus(rvtme.r, moonPosition(rvtme.t)))) + \
             math.sqrt(100 / Vector.absV(Vector.minus(rvtme.r, moonPosition(rvtme.t))))

    # we need to increase our velocity approximately by this value
    tau = rvtme.m / q * (1 - math.exp(-deltaV / u))
    print(deltaV, " ", tau)

    # we need to keep the engine on for approximately this time (according to the Tsiolkovsky equation)
    # -------------------------------------------acceleration-------------------------------------
    rvtme.engine = q
    i = 0
    while rvtme.t < tau:
        rvtme = nextRVTME(rvtme, timestep(acc(rvtme.r, rvtme.v, rvtme.t, rvtme.m, rvtme.engine)))
        output.write(str(rvtme.r.x) + '\t'
                     + str(rvtme.r.y) + '\t'+ str(Vector.absV(rvtme.r)) +
                     '\t' + str(Vector.absV(rvtme.v)) + '\t' + str(rvtme.t) + '\n')
        i += 1
        if i % 10000 == 0:
            print(rvtme.r.x, " ", rvtme.r.y)
    rvtme.engine = 0
    print(Vector.absV(Vector.minus(rvtme.v, moonV(rvtme.t))))
    print(math.sqrt(2 * Gm / Vector.absV(Vector.minus(rvtme.r, moonPosition(rvtme.t)))))
    print(Vector.absV(Vector.minus(rvtme.r, moonPosition(rvtme.t))))

    # -------------------------------------------acceleration-------------------------------------
    # --------------------------------------------waiting for 1 hour------------------------------
    while rvtme.t < 3600:
        rvtme = nextRVTME(rvtme, timestep(acc(rvtme.r, rvtme.v, rvtme.t, rvtme.m, rvtme.engine)))
        output.write(str(rvtme.r.x) + '\t'
                     + str(rvtme.r.y) + '\t' + str(Vector.absV(rvtme.r)) +
                     '\t' + str(Vector.absV(rvtme.v)) + '\t' + str(rvtme.t) + '\n')
        i += 1
        if i % 50000 == 0:
            print(rvtme.r.x, " ", rvtme.r.y)

    # --------------------------------------------waiting for 1 hour-----------------------------
    # --------------------------------------------correction-------------------------------------
    copy = RVTME.copy(rvtme)
    testR = test(copy)
    print(testR)
    while abs(testR - rEarth - 70) > 0.00001:
        copy.v = Vector.mult(1 - 0.0000085 * (rEarth + 70 - testR) / Vector.absV(copy.v), copy.v)
        testR = test(copy)
        print(testR - rEarth)

    print("Reached 1 cm tolerance")
    print("We must increase our velocity by ", 1000 * Vector.absV(Vector.minus(copy.v, rvtme.v)), " m/s")
    rvtme.v = Vector.copy(copy.v)
    # --------------------------------------------correction-------------------------------------------------

    angle = pi / 2 - Vector.angle(rvtme.r, rvtme.v)
    while angle < 0:
        rvtme = nextRVTME(rvtme, timestep(acc(rvtme.r, rvtme.v, rvtme.t, rvtme.m, rvtme.engine), 0.00001))
        angle = pi / 2 - Vector.angle(rvtme.r, rvtme.v)
        i += 1
        if i % 50000 == 0:
            print(rvtme.r.x, " ", rvtme.r.y)
        output.write(str(rvtme.r.x) + '\t' + str(rvtme.r.y) + '\t' + str(Vector.absV(rvtme.r)) +
                         '\t' + str(Vector.absV(rvtme.v)) + '\t' + str(rvtme.t) + '\n')

    print("-----------------------------------")
    print("Finish!")
    print(math.sqrt(rvtme.r.x*rvtme.r.x + rvtme.r.y*rvtme.r.y)-rEarth) #height of our orbit
    print(rvtme.m - dryMass)#check that the fuel is enough
main()

string = open('moontoearth.txt').readlines()
m = array([[float(i) for i in string[k].split()] for k in range((len(string)))])
from matplotlib.pyplot import *
plt.title(' y(x) ', size=11)
plot(list(m[:, 0]/1000), list(m[:, 1]/1000), "blue", markersize=0.1)
plt.xlabel('Coordinate x, km*10^3')
plt.ylabel('Coordinate y, km*10^3')
plt.grid()
show()

plt.title(' r(t) ', size=11)
plot(list(m[:, 4]/1000), list(m[:, 2]/1000), "blue", markersize=0.1)
plt.ylabel('Distance, km*10^3')
plt.xlabel('Time, s*1000')
plt.grid()
show()

plt.title(' V(t) ', size=11)
plot(list(m[:, 4]/1000), list(m[:, 3]), "blue", markersize=0.1)
plt.ylabel('Velocity, km/s ')
plt.xlabel('Time, s*1000')
plt.grid()
show()

