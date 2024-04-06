import numpy as np
from scipy.integrate import solve_bvp
from tqdm import tqdm
from matplotlib import pyplot as plt
π = np.pi


def standing():

    # initialize the params
    n=200 
    loop = tqdm(total=n)
    alpha1, alpha2, alpha3, alpha4 = 0, 9.81, 5*π/180, 7*π/180
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
        #check whether each leg is on the ground
        cond1 = y[1]-(l1*np.cos(y[2])+l2*np.cos(y[4])) == 0
        cond2 = y[1]-(l1*np.cos(y[3])+l2*np.cos(y[5])) == 0

        #make the derivative of the control bangbang
        if y[6]*l1*beta - y[7]*l2*2/π*y[2] < 0:
            dtheta1 = alpha3
        else:
            dtheta1 = -alpha3
        if y[6]*l1*beta - y[7]*l2*2/π*y[3] < 0:
            dtheta2 = alpha3
        else:
            dtheta2 = -alpha3
        if y[6]*l1*beta - y[7]*l2*2/π*y[4] < 0:
            dphi1 = alpha4
        else:
            dphi1 = -alpha4
        if y[6]*l1*beta - y[7]*l2*2/π*y[5] < 0:
            dphi2 = alpha4
        else:
            dphi2 = -alpha4

        # define different parts of the state evolution
        dx1 = l2*dphi1*np.cos(y[4]) + l1*dtheta1*np.cos(y[2])
        dx2 = l2*dphi2*np.cos(y[5]) + l1*dtheta2*np.cos(y[3])
        dy1 = -l2*dphi1*np.sin(y[4]) - l1*dtheta1*np.sin(y[2])
        dy2 = -l2*dphi2*np.sin(y[5]) - l1*dtheta2*np.sin(y[3])
        
        #define the state evolution equations
        if cond1 and cond2: dx, dy = dx1+dx2, dy1+dy2
        elif cond1: dx, dy = dx1, dy1
        elif cond2: dx, dy = dx2, dy2
        else: dx, dy = 0, -alpha2

        val = mu1 - mu2 + mu3 - mu4
        #define how the control evolves
        if cond1:
            u1 = (2*alpha3+l1*dtheta1*y[6])**-1*(val - y[7]*l1*dtheta1*beta)
            u3 = (2*alpha4+l2*dphi1*y[6])**-1*(-mu1+mu2 - y[7]*l2*dphi1*beta)
        else:
            u1 = 1/(2*alpha3)*val
            u3 = 1/(2*alpha4)*(-mu1+mu2)
        if cond2:
            u2 = (2*alpha3+l1*dtheta2*y[6])**-1*(val - y[7]*l1*dtheta2*beta)
            u4 = (2*alpha4+l2*dphi2*y[6])**-1*(-mu1+mu2 - y[7]*l2*dphi2*beta)
        else:
            u2 = 1/(2*alpha3)*val
            u4 = 1/(2*alpha4)*(-mu1+mu2)      
        
        loop.update()
        return np.array([dx, dy, u1, u2, u3, u4, 0, (y[1]+alpha1)**-2])
    
    def bc(ya, yb):
        return np.array(ya[0]-1, ya[1]-1.5, ya[2]+π/90, ya[3]-π/6, ya[4]+π/2, ya[5]+π/2, yb[1]-targety, yb[6], yb[7]+endscale*2*(yb[1]-targety))
    
    #use solve bvp to get a solution
    y0 = np.ones((8,n))
    domain = np.linspace(0, 10, n)
    # p0 = np.array([6]) #try optimizing over tf later
    sol = solve_bvp(ode, bc, domain, y0, max_nodes=60000)
    return sol


if __name__ == '__main__':
    sol = standing()
    domain = np.linspace(0, 10, 200)
    plt.subplot(211)
    plt.title('state')
    plt.plot(sol.y[0], sol.y[1])
    plt.subplot(212)
    plt.title('controls')
    plt.plot(domain, sol.y[2], label = 'θ1')
    plt.plot(domain, sol.y[3], label = 'θ2')
    plt.plot(domain, sol.y[4], label = 'φ1')
    plt.plot(domain, sol.y[5], label = 'φ2')
    plt.legend(fancybox=True, shadow=True)
    plt.tight_layout()
    plt.show()