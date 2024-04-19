import numpy as np
from scipy.integrate import solve_bvp
from matplotlib import pyplot as plt
import animation as ani
π = np.pi


def standing(exp =1):

    # initialize the params
    n=200 #20 fps so 200=10s
    alpha1, alpha2, alpha3, alpha4 = 1, 9.81, (5*π/180)**-1, (5*π/180)**-1
    beta1, beta2, beta3, beta4 = 0.67227, 0.56410, 0.63648, -0.63648
    endscale = 1
    l1, l2 = 1, 1
    targety = 1.8
    targetx = 3
    beta = 0.63598

    def ode(t, y):
        """
        ind   0  1  2   3   4   5   6   7
            ([x, y, θ1, θ2, φ1, φ2, p1, p2])'
        """
        # if True in np.isnan(y):
        #     raise ValueError('NAN found')
        n = len(y[0])
        #defin the complementary slackness conditions
        mu3, mu4, mu7, mu8  = np.ones((4, n))
        mu1, mu2, mu5, mu6  = np.ones((4, n))
        # mu9, mu10, mu11, mu12  = 500*np.ones((4, n)) #these should always be 0
        # mu13, mu14, mu15, mu16  = 500*np.ones((4, n)) #these should always be 0

        mask = y[4]-y[2] == 0
        mu1[~mask] = 0
        mask = -π/2+y[2]-y[4] == 0
        mu2[~mask] = 0
        mask = -π/18 - y[2] == 0
        mu3[~mask] = 0
        mask = -π/2 + y[2] == 0
        mu4[~mask] = 0
        mask = y[5]-y[3] == 0
        mu5[~mask] = 0
        mask = -π/2+y[3]-y[5] == 0
        mu6[~mask] = 0
        mask = -π/18 - y[3] == 0
        mu7[~mask] = 0
        mask = -π/2 + y[3] == 0
        mu8[~mask] = 0

        
        #pre allocate some of the variables
        dtheta1 = np.zeros(n)
        dtheta2 = np.zeros(n)
        dphi1 = np.zeros(n)
        dphi2 = np.zeros(n)
        dx, dy = np.zeros(n), np.zeros(n)
        u1, u2, u3, u4 = np.zeros((4,n))

        #check whether each leg is on the ground
        cond1 = y[1]-(l1*np.cos(y[2])+l2*np.cos(y[4])) <= 0
        cond2 = y[1]-(l1*np.cos(y[3])+l2*np.cos(y[5])) <= 0

        #make the derivative of all 4 controls bangbang
        """
        linearized form:
        mask = y[6]*l1*beta - y[7]*l2*2*(y[2])/π < 0

        mask = y[6]*l1*beta - y[7]*l2*2*(y[3])/π < 0

        mask = y[6]*l1*beta - y[7]*l2*2*(y[4])/π < 0

        mask = y[6]*l1*beta - y[7]*l2*2*(y[5])/π < 0
        """
        mask = y[6]*l1*np.cos(y[2]) - y[7]*l1*np.sin(y[2])< 0
        dtheta1[mask] = alpha3**-1
        dtheta1[~mask] = -alpha3**-1

        mask = y[6]*l1*np.cos(y[3]) - y[7]*l1*np.sin(y[3]) < 0
        dtheta2[mask] = alpha3**-1
        dtheta2[~mask] = -alpha3**-1

        mask = y[6]*l2*np.cos(y[4]) - y[7]*l2*np.sin(y[4]) <0
        dphi1[mask] = alpha4**-1
        dphi1[~mask] = -alpha4**-1

        mask = y[6]*l2*np.cos(y[5]) - y[7]*l2*np.sin(y[5]) < 0
        dphi2[mask] = alpha4**-1
        dphi2[~mask] = -alpha4**-1

        val = mu1 - mu2 + mu3 - mu4
        val2 = mu5 - mu6 + mu7 - mu8
        #define how the control evolves
        u1[cond1] = (2*alpha3+l1*dtheta1[cond1]*y[6][cond1])**-1*(val[cond1] + y[7][cond1]*l1*dtheta1[cond1]*beta)
        u3[cond1] = (2*alpha4+l2*dphi1[cond1]*y[6][cond1])**-1*(-mu1[cond1]+mu2[cond1] + y[7][cond1]*l2*dphi1[cond1]*beta)
        u1[~cond1] = 1/(2*alpha3)*val[~cond1]
        u3[~cond1] = 1/(2*alpha4)*(-mu1[~cond1]+mu2[~cond1])

        u2[cond2] = (2*alpha3+l1*dtheta2[cond2]*y[6][cond2])**-1*(val2[cond2] + y[7][cond2]*l1*dtheta2[cond2]*beta)
        u4[cond2] = (2*alpha4+l2*dphi2[cond2]*y[6][cond2])**-1*(-mu5[cond2]+mu6[cond2] + y[7][cond2]*l2*dphi2[cond2]*beta)
        u2[~cond2] = 1/(2*alpha3)*val2[~cond2]
        u4[~cond2] = 1/(2*alpha4)*(-mu5[~cond2]+mu6[~cond2])  

        # define different parts of the state evolution
        dx1 = l2*dphi1*np.cos(u3) + l1*dtheta1*np.cos(u1)
        dx2 = l2*dphi2*np.cos(u4) + l1*dtheta2*np.cos(u2)
        dy1 = (l2*dphi1*np.sin(u3) + l1*dtheta1*np.sin(u1))
        dy2 = (l2*dphi2*np.sin(u4) + l1*dtheta2*np.sin(u2))
        
        #define the state evolution equations
        for i in range(n):
            if cond1[i] and cond2[i]: dx[i], dy[i] = dx1[i]+dx2[i], dy1[i]+dy2[i] ###### changed all x' and y' signs to - didn't change anything but may want to leave in 
            elif cond1[i]: dx[i], dy[i] = dx1[i], dy1[i]
            elif cond2[i]: dx[i], dy[i] = dx2[i], dy2[i]
            else: dx[i], dy[i] = 0, -alpha2
           
        
        return np.array([dx, dy, u1, u2, u3, u4, np.zeros(n), 4*(alpha1*y[1]-alpha1*targety)**3])
    
    """bc i've tested:
    return np.array([ya[0]-1, ya[1]-1.5, ya[2]+π/90, ya[3]-π/6, ya[4]+π/2, ya[5]+π/2, yb[1]-targety, yb[6], yb[7]+endscale*2*(yb[1]-targety)])
    return np.array([ya[0]-1, ya[1]-1.5, ya[2]+π/90, ya[3]-π/6, ya[4]+π/2, ya[5]+π/2, yb[6], yb[7]+endscale*2*(yb[1]-targety)])
    ^this one 'worked' gave me something interesting but went below the x axis

    """
    def bc(ya, yb):
        return np.array([ya[0]-1, ya[1]-1.5, ya[2]+π/90, ya[3]-π/6, ya[4]+π/2, ya[5]+π/2, yb[6]-2*endscale*targetx, yb[7]-2*endscale*targety])
    
    def ode2(t, y):
        """
        ind   0  1  2   3   4   5   6   7
            ([x, y, θ1, θ2, φ1, φ2, p1, p2])'
        """
        # if True in np.isnan(y):
        #     raise ValueError('NAN found')
        n = len(y[0])
        #defin the complementary slackness conditions
        mu3, mu4, mu7, mu8  = np.ones((4, n))
        mu1, mu2, mu5, mu6  = np.ones((4, n))

        mask = y[4]-y[2] == 0
        mu1[~mask] = 0
        mask = -π/2+y[2]-y[4] == 0
        mu2[~mask] = 0
        mask = -π/18 - y[2] == 0
        mu3[~mask] = 0
        mask = -π/2 + y[2] == 0
        mu4[~mask] = 0
        mask = y[5]-y[3] == 0
        mu5[~mask] = 0
        mask = -π/2+y[3]-y[5] == 0
        mu6[~mask] = 0
        mask = -π/18 - y[3] == 0
        mu7[~mask] = 0
        mask = -π/2 + y[3] == 0
        mu8[~mask] = 0

        
        #pre allocate some of the variables
        dtheta1 = np.zeros(n)
        dtheta2 = np.zeros(n)
        dphi1 = np.zeros(n)
        dphi2 = np.zeros(n)
        dx, dy = np.zeros(n), np.zeros(n)
        u1, u2, u3, u4 = np.zeros((4,n))

        #check whether each leg is on the ground
        cond1 = y[1]-(l1*np.cos(y[2])+l2*np.cos(y[4])) <= 0
        cond2 = y[1]-(l1*np.cos(y[3])+l2*np.cos(y[5])) <= 0

        #make the derivative of all 4 controls bangbang
        """
        linearized form:
        mask = y[6]*l1*beta - y[7]*l2*2*(y[2])/π < 0

        mask = y[6]*l1*beta - y[7]*l2*2*(y[3])/π < 0

        mask = y[6]*l1*beta - y[7]*l2*2*(y[4])/π < 0

        mask = y[6]*l1*beta - y[7]*l2*2*(y[5])/π < 0
        """
        mask = y[6]*l1*np.cos(y[2]) - y[7]*l1*np.sin(y[2]) < 0
        dtheta1[mask] = alpha3**-1
        dtheta1[~mask] = -alpha3**-1

        mask = y[6]*l1*np.cos(y[3]) - y[7]*l1*np.sin(y[3]) < 0
        dtheta2[mask] = alpha3**-1
        dtheta2[~mask] = -alpha3**-1

        mask = y[6]*l2*np.cos(y[4]) - y[7]*l2*np.sin(y[4]) < 0
        dphi1[mask] = alpha4**-1
        dphi1[~mask] = -alpha4**-1

        mask = y[6]*l2*np.cos(y[5]) - y[7]*l2*np.sin(y[5]) < 0
        dphi2[mask] = alpha4**-1
        dphi2[~mask] = -alpha4**-1

        # define different parts of the state evolution
        dx1 = l2*dphi1*np.cos(y[4]) + l1*dtheta1*np.cos(y[2])
        dx2 = l2*dphi2*np.cos(y[5]) + l1*dtheta2*np.cos(y[3])
        dy1 = l2*dphi1*np.sin(y[4]) + l1*dtheta1*np.sin(y[2])
        dy2 = l2*dphi2*np.sin(y[5]) + l1*dtheta2*np.sin(y[3])
        
        #define the state evolution equations
        for i in range(n):
            if cond1[i] and cond2[i]: dx[i], dy[i] = dx1[i]+dx2[i], dy1[i]+dy2[i] ###### changed all x' and y' signs to - didn't change anything but may want to leave in 
            elif cond1[i]: dx[i], dy[i] = dx1[i], dy1[i]
            elif cond2[i]: dx[i], dy[i] = dx2[i], dy2[i]
            else: dx[i], dy[i] = 0, -alpha2

        val = mu1 - mu2 + mu3 - mu4
        val2 = mu5 - mu6 + mu7 - mu8
        #define how the control evolves
        u1[cond1] = -np.log(y[6][cond1]*l1*np.sin(y[2][cond1])-y[7][cond1]*l1*np.cos(y[2][cond1]))+np.log(-2*alpha3*y[2][cond1]+val[cond1])
        u3[cond1] = -np.log(y[6][cond1]*l1*np.sin(y[4][cond1])-y[7][cond1]*l1*np.cos(y[4][cond1]))+np.log(-2*alpha3*y[4][cond1]-mu1[cond1]+mu2[cond1])
        u1[~cond1] = -np.log(alpha3)
        u3[~cond1] = -np.log(alpha4)

        u2[cond2] = -np.log(y[6][cond2]*l2*np.sin(y[3][cond2])-y[7][cond2]*l2*np.cos(y[3][cond2]))+np.log(-2*alpha3*y[3][cond2]+val2[cond2])
        u4[cond2] = -np.log(y[6][cond2]*l2*np.sin(y[5][cond2])-y[7][cond2]*l2*np.cos(y[5][cond2]))+np.log(-2*alpha3*y[5][cond2]-mu5[cond2]+mu6[cond2])
        u2[~cond2] = -np.log(alpha3)
        u4[~cond2] = -np.log(alpha3)     
        
        return np.array([dx, dy, np.exp(u1), np.exp(u2), np.exp(u3), np.exp(u4), np.zeros(n), (alpha1*y[1]-alpha1*targety)**3])

    def ode3(t, y):
        """
        ind   0  1  2   3   4   5   6    7    8    9    10  11
            ([x, y, θ1, θ2, φ1, φ2, dθ1, dθ2, dφ1, dφ2, p1, p2])'
        """
        # if True in np.isnan(y):
        #     raise ValueError('NAN found')
        n = len(y[0])
        #defin the complementary slackness conditions
        mu3, mu4, mu7, mu8  = np.ones((4, n))
        mu1, mu2, mu5, mu6  = np.ones((4, n))

        mask = y[4]-y[2] == 0
        mu1[~mask] = 0
        mask = -π/2+y[2]-y[4] == 0
        mu2[~mask] = 0
        mask = -π/18 - y[2] == 0
        mu3[~mask] = 0
        mask = -π/2 + y[2] == 0
        mu4[~mask] = 0
        mask = y[5]-y[3] == 0
        mu5[~mask] = 0
        mask = -π/2+y[3]-y[5] == 0
        mu6[~mask] = 0
        mask = -π/18 - y[3] == 0
        mu7[~mask] = 0
        mask = -π/2 + y[3] == 0
        mu8[~mask] = 0

        
        #pre allocate some of the variables
        dx, dy = np.zeros(n), np.zeros(n)
        dtheta1 = np.zeros(n)
        dtheta2 = np.zeros(n)
        dphi1 = np.zeros(n)
        dphi2 = np.zeros(n)

        #check whether each leg is on the ground
        cond1 = y[1]-(l1*np.cos(y[2])+l2*np.cos(y[4])) <= 0
        cond2 = y[1]-(l1*np.cos(y[3])+l2*np.cos(y[5])) <= 0
        val = mu1 - mu2 + mu3 - mu4
        val2 = mu5 - mu6 + mu7 - mu8
        #define how the control evolves
        dtheta1[cond1] = -np.log((y[10][cond1]*l1*np.sin(y[2][cond1])-y[11][cond1]*l1*np.cos(y[2][cond1])))+np.log((-2*alpha3*y[2][cond1]+val[cond1]))
        dtheta1[~cond1] = -np.log(alpha3)
        dtheta2[cond2] = -np.log((y[10][cond2]*l1*np.sin(y[3][cond2])-y[11][cond2]*l1*np.cos(y[3][cond2])))+np.log((-2*alpha3*y[3][cond2]+val2[cond2]))
        dtheta2[~cond2] = -np.log(alpha3)
        dphi1[cond1] = -np.log((y[10][cond1]*l2*np.sin(y[4][cond1])-y[11][cond1]*l2*np.cos(y[4][cond1])))+np.log((-2*alpha3*y[4][cond1]-mu1[cond1]+mu2[cond1]))
        dphi1[~cond1] = -np.log(alpha4)
        dphi2[cond2] = -np.log((y[10][cond2]*l2*np.sin(y[5][cond2])-y[11][cond2]*l2*np.cos(y[5][cond2])))+np.log((-2*alpha3*y[5][cond2]-mu5[cond2]+mu6[cond2]))
        dphi2[~cond2] = -np.log(alpha4)
        dtheta1 = np.exp(dtheta1)
        dtheta2 = np.exp(dtheta2)
        dphi1 = np.exp(dphi1)
        dphi2 = np.exp(dphi2)
        """LOG BASE 2 VERSION:
        dtheta1[cond1] = -np.log2((y[10][cond1]*l1*np.sin(y[2][cond1])-y[11][cond1]*l1*np.cos(y[2][cond1])))+np.log2((-2*alpha3*y[2][cond1]+val[cond1]))
        dtheta1[~cond1] = -np.log2(alpha3)
        dtheta2[cond2] = -np.log2((y[10][cond2]*l1*np.sin(y[3][cond2])-y[11][cond2]*l1*np.cos(y[3][cond2])))+np.log2((-2*alpha3*y[3][cond2]+val2[cond2]))
        dtheta2[~cond2] = -np.log2(alpha3)
        dphi1[cond1] = -np.log2((y[10][cond1]*l2*np.sin(y[4][cond1])-y[11][cond1]*l2*np.cos(y[4][cond1])))+np.log2((-2*alpha3*y[4][cond1]-mu1[cond1]+mu2[cond1]))
        dphi1[~cond1] = -np.log2(alpha4)
        dphi2[cond2] = -np.log2((y[10][cond2]*l2*np.sin(y[5][cond2])-y[11][cond2]*l2*np.cos(y[5][cond2])))+np.log2((-2*alpha3*y[5][cond2]-mu5[cond2]+mu6[cond2]))
        dphi2[~cond2] = -np.log2(alpha4)
        dtheta1 = (dtheta1)**2
        dtheta2 = (dtheta2)**2
        dphi1 = (dphi1)**2
        dphi2 = (dphi2)**2
        """

        # define different parts of the state evolution
        dx1 = l2*dphi1*np.cos(y[4]) + l1*dtheta1*np.cos(y[2])
        dx2 = l2*dphi2*np.cos(y[5]) + l1*dtheta2*np.cos(y[3])
        dy1 = (l2*dphi1*np.sin(y[4]) + l1*dtheta1*np.sin(y[2]))
        dy2 = (l2*dphi2*np.sin(y[5]) + l1*dtheta2*np.sin(y[3]))
        
        #define the state evolution equations
        for i in range(n):
            if cond1[i] and cond2[i]: dx[i], dy[i] = dx1[i]+dx2[i], dy1[i]+dy2[i] ###### changed all x' and y' signs to - didn't change anything but may want to leave in 
            elif cond1[i]: dx[i], dy[i] = dx1[i], dy1[i]-0.4*alpha2
            elif cond2[i]: dx[i], dy[i] = dx2[i], dy2[i]-0.4*alpha2
            else: dx[i], dy[i] = 0, -alpha2
        
        return np.array([dx, dy, y[6], y[7], y[8], y[9], dtheta1, dtheta2, dphi1, dphi2, np.zeros(n), 4*(alpha1*y[1]-alpha1*targety)**3])
    
    def bc3(ya, yb):
        """
        ind   0  1  2   3   4   5   6    7    8    9    10  11
            ([x, y, θ1, θ2, φ1, φ2, dθ1, dθ2, dφ1, dφ2, p1, p2])'
            sets of conditions i've tried
            [ya[0]-1, ya[1]-1.5, ya[2]+π/90, ya[3]-π/6, ya[4]+π/2, ya[5]+π/2, ya[8]-alpha4**-1, ya[9]-alpha4**-1, yb[0]-targetx, yb[1]-targety, yb[10]+2*endscale*targetx, yb[11]+2*endscale*targety]
            [ ya[2]+π/90, ya[3]-π/6, ya[4]+π/2, ya[5]+π/2, ya[6]-alpha3**-1, ya[7]-alpha3**-1, ya[8]-alpha4**-1, ya[9]-alpha4**-1, yb[0]-targetx, yb[1]-targety, yb[10]+2*endscale*targetx, yb[11]+2*endscale*targety]
        """
        return np.array([ ya[2]+π/90, ya[3]-π/6, ya[4]+π/2, ya[5]+π/2, ya[6]-alpha3**-1, ya[7]-alpha3**-1, ya[8]-alpha4**-1, ya[9]-alpha4**-1, yb[0]-targetx, yb[1]-targety, yb[10]+2*endscale*targetx, yb[11]+2*endscale*targety])

    #use solve bvp to get a solution
    y0 = np.ones((8,n))
    y0[0] = y0[0]
    y0[1] = y0[1]*1.8
    y0[2] = y0[2]*-π/90
    y0[3] = y0[3]*π/6
    y0[4] = np.linspace(-π/2, 0, n)
    y0[5] = np.linspace(-π/2, 0, n)
    # y0[6] = y0[6]*alpha3**-1
    # y0[7] = y0[7]*alpha3**-1
    # y0[8] = y0[8]*alpha4**-1
    # y0[9] = y0[9]*alpha4**-1
    domain = np.linspace(0, 10, n)
    # p0 = np.array([6]) #try optimizing over tf later
    sol = solve_bvp(ode2, bc, domain, y0, max_nodes=30000, tol=1e-6)
    return sol

def step1():

    # initialize the params
    n=200 #20 fps so 200=10s

    alpha1, alpha3, alpha4 = 1, (5*π/180), (5*π/180)
    endscale = 1 # lambda
    l1, l2 = 1, 1
    targety = 1.8
    targetx = 3
    beta = 0.63598
    l3 = 1.8

    def ode(t, y):
        """
        ind   0  1  2  3  4  5   6   7   8   9
            ([x, y, θ, φ, ψ, dθ, dφ, p1, p2, p3])'
        """
        n = len(y[0])
        #initialize the lagrange mults for the ineq. constraints
        mu3, mu4 = np.ones((2, n))
        mu1, mu2  = np.ones((2, n))

        # use complemmentary slackness
        mask = y[3]-y[2] == 0
        mu1[~mask] = 0
        mask = -π/2+y[2]-y[3] == 0
        mu2[~mask] = 0
        mask = -π/18 - y[2] == 0
        mu3[~mask] = 0
        mask = -π/2 + y[2] == 0
        mu4[~mask] = 0
        
        #pre allocate some of the variables
        dx, dy = np.zeros(n), np.zeros(n)
        dtheta = np.zeros(n)
        dphi = np.zeros(n)

        #check whether each leg is on the ground
        cond1 = y[1]-(l1*np.cos(y[2])+l2*np.cos(y[3])) <= 0
        val = mu1 - mu2 + mu3 - mu4
        #define how the control evolves
        dtheta[cond1] = -np.log((y[7][cond1]*l1*np.sin(y[2][cond1])-y[8][cond1]*l1*np.cos(y[2][cond1])))+np.log((-2*(alpha3)**-1*y[2][cond1]+val[cond1]))
        dtheta[~cond1] = np.log(alpha3)
        dphi[cond1] = -np.log((y[7][cond1]*l2*np.sin(y[3][cond1])-y[8][cond1]*l2*np.cos(y[3][cond1])))+np.log((-2*(alpha4)**-1*y[3][cond1]-mu1[cond1]+mu2[cond1]))
        dphi[~cond1] = np.log(alpha4)
        dtheta = np.exp(dtheta)
        dphi = np.exp(dphi)

        # define different parts of the state evolution
        dx[cond1] = l2*dphi[cond1]*np.cos(y[3][cond1]) + l1*dtheta[cond1]*np.cos(y[2][cond1])
        dx[~cond1] = l3*alpha3*np.cos(y[4][~cond1])
        dy[cond1] = (l2*dphi[cond1]*np.sin(y[3][cond1]) + l1*dtheta[cond1]*np.sin(y[2][cond1]))
        dx[~cond1] = l3*alpha3*np.sin(y[4][~cond1])
        
        p3 = y[7]*(-l3*alpha3**2*np.sin(y[4]))+y[8]*(-l3*alpha3**2*np.sin(y[4]))
        
        return np.array([dx, dy, y[5], y[6], alpha3*np.ones(n), dtheta, dphi, np.zeros(n), 4*(alpha1**-1*(y[1]-targety))**3, p3])
    
    #define the boundary conditions
    def bc(ya, yb):
        """
        ind   0  1  2  3  4  5   6   7   8   9
            ([x, y, θ, φ, ψ, dθ, dφ, p1, p2, p3])'
        """
        return np.array([ya[2], ya[3]-π/6, ya[4]+π/4, ya[5]-alpha3, ya[6]-alpha3, yb[0]-targetx, yb[1]-targety, yb[7]+2*endscale*targetx, yb[8], yb[9]])


    #use solve bvp to get a solution
    y0 = np.ones((10,n))
    y0[0] = np.linspace(-1, 1, n)
    y0[1] = y0[1]*1.8
    y0[2] = y0[2]*-π/90
    y0[4] = np.linspace(-π/4, π/4, n)
    domain = np.linspace(0, 8, n)
    # p0 = np.array([6]) #try optimizing over tf later
    sol = solve_bvp(ode, bc, domain, y0, max_nodes=30000, tol=1e-6)
    return sol



if __name__ == '__main__':
    # sol = standing()
    # n = len(sol.y[0])
    # domain = np.linspace(0, 10, n)
    # plt.subplot(211)
    # plt.title('state')
    # plt.scatter([sol.y[0][0], sol.y[0][-1]], [sol.y[1][0], sol.y[1][-1]], c=['r', 'g'])
    # plt.subplot(212)
    # plt.title('controls')
    # plt.plot(domain, sol.y[2], label = 'θ1')
    # plt.plot(domain, sol.y[3], label = 'θ2')
    # plt.plot(domain, sol.y[4], label = 'φ1')
    # plt.plot(domain, sol.y[5], label = 'φ2')
    # plt.legend(fancybox=True, shadow=True)
    # plt.tight_layout()
    # plt.show(block=True)
    # ani.JaAdLiTy_learns_to_walk(np.array((sol.y[0], sol.y[1])), np.array((sol.y[2], sol.y[3], sol.y[4], sol.y[5])), (1, 1), 5)
    # print(sol.y[0], sol.y[1])

    sol = step1()
    n = len(sol.y[0])
    domain = np.linspace(0, 10, n)
    plt.subplot(211)
    plt.title('state (start and end)')
    plt.scatter([sol.y[0][0], sol.y[0][-1]], [sol.y[1][0], sol.y[1][-1]], c=['g', 'r'])
    plt.subplot(212)
    plt.title('controls')
    plt.plot(domain, sol.y[2], label = 'θ')
    plt.plot(domain, sol.y[3], label = 'φ')
    plt.legend(fancybox=True, shadow=True)
    plt.tight_layout()
    plt.show(block=True)
    # print(sol.y[0], sol.y[1])