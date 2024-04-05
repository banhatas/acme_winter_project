import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# animation.writer = animation.writers['ffmpeg']

def JaAdLiTy_learns_to_walk(state, control, leg_length, endpoint):
    """
    state: ndarray of position of mass at each timestep.
    control: ndarray of all four angles in radians of the leg at each timestep.
    leg_length: tuple (int,int) of the length of the legs.
    endpoint: int of end point.
    TODO: add in terrain line?
    """
    # Setup frames and interval
    T = len(state)
    ts = list(range(T))
    interval = np.ceil(1000/(T - 1))

    # Setup graph
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.xlim([-.1, endpoint + 1])
    plt.ylim([-.1, sum(leg_length) + 1])
    plt.axis('off')

    # Setup the lines and points
    ground, = ax.plot([], [], 'k-')
    thigh1, = ax.plot([], [], 'b-')
    thigh2, = ax.plot([], [], 'g-')
    calf1, = ax.plot([], [], 'b-')
    calf2, = ax.plot([], [], 'g-')
    mass, = ax.plot([], [], 'r.')

    # Create the update function
    def update(t):
        t = t//10
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

        # Determine the slope of each part of the legs
        slope_t1 = (B1[1] - A[1])/(B1[0] - A[0]) if (B1[0] - A[0]) != 0 else (B1[1] - A[1])
        slope_t2 = (B2[1] - A[1])/(B2[0] - A[0]) if (B2[0] - A[0]) != 0 else (B2[1] - A[1])
        slope_c1 = (C1[1] - B1[1])/(C1[0] - B1[0]) if (C1[0] - B1[0]) != 0 else (C1[1] - B1[1])
        slope_c2 = (C2[1] - B2[1])/(C2[0] - B2[0]) if (C2[0] - B2[0]) != 0 else (C2[1] - B2[1])

        # Create equations for the lines of the legs
        t1 = lambda x: slope_t1*x + A[1] - slope_t1*A[0]
        t2 = lambda x: slope_t2*x + A[1] - slope_t2*A[0]
        c1 = lambda x: slope_c1*x + B1[1] - slope_c1*B1[0]
        c2 = lambda x: slope_c2*x + B2[1] - slope_c2*B2[0]

        # Setup the domain for the legs
        x_t1 = np.linspace(A[0],B1[0],100)
        x_c1 = np.linspace(B1[0],C1[0],100)
        x_t2 = np.linspace(A[0],B2[0],100)
        x_c2 = np.linspace(B2[0],C2[0],100)

        # Update the animation
        ground.set_data([0,endpoint],[0,0])
        thigh1.set_data(x_t1, t1(x_t1))
        thigh2.set_data(x_t2, t2(x_t2))
        calf1.set_data(x_c1, c1(x_c1))
        calf2.set_data(x_c2, c2(x_c2))
        mass.set_data(A[0],A[1])

    ani = animation.FuncAnimation(fig, update, frames=(T-1)*100, interval=100)
    ani.save('JaAdLiTy_walks.mp4')

if __name__ == '__main__':
    # Test case
    state = np.array([[5,5,5,5,5,5,5,5,5,5],       # x values
                      [8,8,8,8,8,8,8,8,8,8]])      # y values
    control = np.array([[-np.pi/2, -np.pi/3, -np.pi/4, -np.pi/6, 0, 0, np.pi/6, np.pi/4, np.pi/3, np.pi/2],     # theta1 values
                        [np.pi/2, np.pi/3, np.pi/4, np.pi/6, 0, 0, -np.pi/6, -np.pi/4, -np.pi/3, -np.pi/2],     # theta2 values
                        [-2*np.pi/3, -np.pi/2, -np.pi/3, -np.pi/4, -np.pi/6, -np.pi/6, -np.pi/4, -np.pi/3, -np.pi/2, -2*np.pi/3],     # phi1 values
                        [0, -np.pi/6, -np.pi/4, -np.pi/3, -np.pi/2, -np.pi/2, -np.pi/3, -np.pi/4, -np.pi/6, 0]])    # phi2 values
    leg_length = (5,3)
    endpoint = 10

    JaAdLiTy_learns_to_walk(state,control,leg_length,endpoint)