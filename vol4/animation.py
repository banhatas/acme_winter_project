import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def JaAdLiTy_learns_to_walk(state, control, leg_length, endpoint):
    """
    state: ndarray of position of mass at each timestep.
    control: ndarray of all four angles of the leg at each timestep.
    leg_length: tuple (int,int) of the length of the legs.
    endpoint: int of end point.
    TODO: add in terrain line?
    """
    T = len(state)
    ts = np.linspace(0,10,T)
    interval = np.ceil(1000*(10)/(T - 1))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.xlim([-1, endpoint + 1])
    plt.ylim([-.1, sum(leg_length) + 1])

    mass, = ax.plot([], [], 'b.')
    thigh1, = ax.plot([], [], 'k-')
    thigh2, = ax.plot([], [], 'r-')
    calf1, = ax.plot([], [], 'k-')
    calf2, = ax.plot([], [], 'r-')

    def update(t):
        theta1 = control[0,t]
        theta2 = control[1,t]
        phi1 = control[2,t]
        phi2 = control[3,t]

        A = (state[0,t], state[1,t])
        B1 = (A[0] + leg_length[0]*np.sin(theta1), A[1] - leg_length[0]*np.cos(theta1))
        B2 = (A[0] + leg_length[0]*np.sin(theta2), A[1] - leg_length[0]*np.cos(theta2))
        C1 = (B1[0] + leg_length[0]*np.sin(phi1+theta1), B1[1] - leg_length[0]*np.cos(phi1+theta1))
        C2 = (B2[0] + leg_length[0]*np.sin(phi2+theta2), B2[1] - leg_length[0]*np.cos(phi2+theta2))

        slope_t1 = (B1[1] - A[1])/(B1[0] - A[0])
        slope_t2 = (B2[1] - A[1])/(B2[0] - A[0])
        slope_c1 = (C1[1] - B1[1])/(C1[0] - B1[0])
        slope_c2 = (C2[1] - B2[1])/(C2[0] - B2[0])

        t1 = lambda x: slope_t1*x + A[1] - slope_t1*A[0]
        t2 = lambda x: slope_t2*x + A[1] - slope_t2*A[0]
        c1 = lambda x: slope_c1*x + B1[1] - slope_c1*B1[0]
        c2 = lambda x: slope_c2*x + B2[1] - slope_c2*B2[0]

        x_t1 = np.linspace(A[0],B1[0],100)
        x_c1 = np.linspace(B1[0],C1[0],100)
        x_t2 = np.linspace(A[0],B2[0],100)
        x_c2 = np.linspace(B2[0],C2[0],100)

        mass.set_data(A[0],A[1])
        thigh1.set_data(x_t1, t1(x_t1))
        thigh2.set_data(x_t2, t2(x_t2))
        calf1.set_data(x_c1, c1(x_c1))
        calf2.set_data(x_c2, c2(x_c2))

    ani = animation.FuncAnimation(fig, update, frames=ts, interval=interval)
    ani.save('JaAdLiTy_walks.mp4')