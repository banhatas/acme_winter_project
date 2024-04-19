import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def JaAdLiTy_learns_to_walk(state, control, leg_length, endpoint):
    """
    state: ndarray of position of mass at each timestep.
    control: ndarray of all four angles in radians of the leg at each timestep.
    leg_length: tuple (int,int) of the length of the legs.
    endpoint: int of end point.
    """
    # Setup frames and interval
    T = len(state)
    ts = list(range(T))
    interval = np.ceil(1000/(T - 1))

    # Setup graph
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.xlim([-.1 - sum(leg_length), endpoint + sum(leg_length) + 1])
    plt.ylim([-.1, 2*sum(leg_length)])
    # plt.axis('off')
    ax.axis('equal')

    # Setup the lines and points
    ground, = ax.plot([], [], 'k-')
    thigh1, = ax.plot([], [], 'b-')
    thigh2, = ax.plot([], [], 'g-')
    calf1, = ax.plot([], [], 'b-')
    calf2, = ax.plot([], [], 'g-')
    knee1, = ax.plot([], [], 'k.')
    knee2, = ax.plot([], [], 'k.')
    foot1, = ax.plot([], [], 'r.')
    foot2, = ax.plot([], [], 'r.')
    mass, = ax.plot([], [], 'r.')

    # Create the update function
    def update(t):
        # Define the angles from the control
        theta1 = control[0,t]
        theta2 = control[1,t]
        phi1 = control[2,t]
        phi2 = control[3,t]

        # Determine the coordinates of the mass, knees, and feet
        A = (state[0,t], state[1,t])
        B1 = (A[0] + leg_length[0]*np.sin(theta1), A[1] - leg_length[0]*np.cos(theta1))
        B2 = (A[0] + leg_length[0]*np.sin(theta2), A[1] - leg_length[0]*np.cos(theta2))
        C1 = (B1[0] + leg_length[1]*np.sin(phi1+theta1), B1[1] - leg_length[1]*np.cos(phi1+theta1))
        C2 = (B2[0] + leg_length[1]*np.sin(phi2+theta2), B2[1] - leg_length[1]*np.cos(phi2+theta2))
        # print(np.sqrt((A[0] - B1[0])**2 + (A[1] - B1[1])**2))
        
        ground.set_data([0,endpoint],[0,0])
        thigh1.set_data([A[0],B1[0]],[A[1],B1[1]])
        thigh2.set_data([A[0],B2[0]],[A[1],B2[1]])
        calf1.set_data([C1[0],B1[0]],[C1[1],B1[1]])
        calf2.set_data([C2[0],B2[0]],[C2[1],B2[1]])
        knee1.set_data(B1[0],B1[1])
        knee2.set_data(B2[0],B2[1])
        foot1.set_data(C1[0],C1[1])
        foot2.set_data(C2[0],C2[1])

        # Update the animation
        mass.set_data(A[0],A[1])

    ani = animation.FuncAnimation(fig, update, frames=(T-1)*200, interval=15)
    ani.save('JaAdLiTy_walks.mp4')