import numpy as np
from scipy.integrate import solve_bvp
from tqdm import tqdm
from matplotlib import pyplot as plt
π = np.pi


def standing():

    # initialize the params
    n=200 #20 fps so 200=10s
    loop = tqdm(total=n)
    alpha1, alpha2, alpha3, alpha4 = 5, 9.81, 5*π/180, 7*π/180
    endscale = 1
    mu1, mu2, mu3, mu4 = 1, 1, 1, 1
    l1, l2 = 1, 1
    targety = 1.8
    targetx = 1
    beta = 0.63598

    def ode(t, y):
        """
        ind   0  1  2   3   4   5   6   7
            ([x, y, θ1, θ2, φ1, φ2, p1, p2])'
        """
        n = len(y[0])
        #pre allocate some of the variables
        dtheta1 = np.zeros(n)
        dtheta2 = np.zeros(n)
        dphi1 = np.zeros(n)
        dphi2 = np.zeros(n)
        dx, dy = np.zeros(n), np.zeros(n)
        u1, u2, u3, u4 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

        #check whether each leg is on the ground
        cond1 = y[1]-(l1*np.cos(y[2])+l2*np.cos(y[4])) == 0
        cond2 = y[1]-(l1*np.cos(y[3])+l2*np.cos(y[5])) == 0

        #make the derivative of all 4 controls bangbang
        mask = y[6]*l1*beta - y[7]*l2*2/π*y[2] < 0
        dtheta1[mask] = alpha3
        dtheta1[~mask] = -alpha3

        mask = y[6]*l1*beta - y[7]*l2*2/π*y[3] < 0
        dtheta2[mask] = alpha3
        dtheta2[~mask] = -alpha3

        mask = y[6]*l1*beta - y[7]*l2*2/π*y[4] < 0
        dphi1[mask] = alpha4
        dphi1[~mask] = -alpha4

        mask = y[6]*l1*beta - y[7]*l2*2/π*y[5] < 0
        dphi2[mask] = alpha4
        dphi2[~mask] = -alpha4

        # define different parts of the state evolution
        dx1 = l2*dphi1*np.cos(y[4]) + l1*dtheta1*np.cos(y[2])
        dx2 = l2*dphi2*np.cos(y[5]) + l1*dtheta2*np.cos(y[3])
        dy1 = -l2*dphi1*np.sin(y[4]) - l1*dtheta1*np.sin(y[2])
        dy2 = -l2*dphi2*np.sin(y[5]) - l1*dtheta2*np.sin(y[3])
        
        #define the state evolution equations
        for i in range(n):
            if cond1[i] and cond2[i]: dx[i], dy[i] = -dx1[i]-dx2[i], -dy1[i]-dy2[i] ###### changed all x' and y' signs to - didn't change anything but may want to leave in
            elif cond1[i]: dx[i], dy[i] = -dx1[i], -dy1[i]
            elif cond2[i]: dx[i], dy[i] = -dx2[i], -dy2[i]
            else: dx[i], dy[i] = 0, -alpha2

        val = mu1 - mu2 + mu3 - mu4
        #define how the control evolves
        u1[cond1] = (2*alpha3+l1*dtheta1[cond1]*y[6][cond1])**-1*(val - y[7][cond1]*l1*dtheta1[cond1]*beta)
        u3[cond1] = (2*alpha4+l2*dphi1[cond1]*y[6][cond1])**-1*(-mu1+mu2 - y[7][cond1]*l2*dphi1[cond1]*beta)
        u1[~cond1] = 1/(2*alpha3)*val
        u3[~cond1] = 1/(2*alpha4)*(-mu1+mu2)

        u2[cond2] = (2*alpha3+l1*dtheta2[cond2]*y[6][cond2])**-1*(val - y[7][cond2]*l1*dtheta2[cond2]*beta)
        u4[cond2] = (2*alpha4+l2*dphi2[cond2]*y[6][cond2])**-1*(-mu1+mu2 - y[7][cond2]*l2*dphi2[cond2]*beta)
        u2[~cond2] = 1/(2*alpha3)*val
        u4[~cond2] = 1/(2*alpha4)*(-mu1+mu2)      
        
        loop.update()
        return np.array([dx, dy, u1, u2, u3, u4, np.zeros(n), 4*(alpha1*y[1]-alpha1*targety)**3])
    
    """bc i've tested:
    return np.array([ya[0]-1, ya[1]-1.5, ya[2]+π/90, ya[3]-π/6, ya[4]+π/2, ya[5]+π/2, yb[1]-targety, yb[6], yb[7]+endscale*2*(yb[1]-targety)])
    return np.array([ya[0]-1, ya[1]-1.5, ya[2]+π/90, ya[3]-π/6, ya[4]+π/2, ya[5]+π/2, yb[6], yb[7]+endscale*2*(yb[1]-targety)])
    ^this one 'worked' gave me something interesting but went below the x axis

    """
    def bc(ya, yb):
        return np.array([ya[0]-1, ya[1]-1.5, ya[2]+π/90, ya[3]-π/6, ya[4]+π/2, ya[5]+π/2, yb[6], yb[7]])
    
    #use solve bvp to get a solution
    y0 = np.ones((8,n))
    domain = np.linspace(0, 10, n)
    # p0 = np.array([6]) #try optimizing over tf later
    sol = solve_bvp(ode, bc, domain, y0, max_nodes=60000)
    return sol


if __name__ == '__main__':
    sol = standing()
    n = len(sol.y[0])
    domain = np.linspace(0, 10, n)
    plt.subplot(211)
    plt.title('state')
    plt.scatter([sol.y[0][0], sol.y[0][-1]], [sol.y[0][1], sol.y[1][-1]], c=['r', 'g'])
    plt.subplot(212)
    plt.title('controls')
    plt.plot(domain, sol.y[2], label = 'θ1')
    plt.plot(domain, sol.y[3], label = 'θ2')
    plt.plot(domain, sol.y[4], label = 'φ1')
    plt.plot(domain, sol.y[5], label = 'φ2')
    plt.legend(fancybox=True, shadow=True)
    plt.tight_layout()
    plt.show()