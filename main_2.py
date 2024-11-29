import numpy as np
import mujoco
import mujoco.viewer
import time
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path(f"snake_3.xml")
data = mujoco.MjData(model)

time_step = model.opt.timestep
head_x = []
head_y = []
theta = 0

def move_snake(amp=0.3, freq=0.5, phase=-np.pi/2 ):
    global theta
    p = np.zeros(5)
    theta += freq * time_step
    for i in range(5):
        p[i] = np.clip(amp * np.sin(theta + i * phase), -3, 3)
        # if i == 0:
            # p[i] = controller(data,  i+1)
        # q[i] = controller(data, float(p[i]), i+1)
    return p

def move_circle(amp=0.3, freq=1, phase=-np.pi/2, offset=np.pi/10):
    global theta, data
    p = np.zeros(5)
    theta += freq * time_step
    for i in range(5):
        p[i] = np.clip(amp * np.sin(theta + i * phase) + offset, -1, 1)
    return p

def controller(model_data: object, actuator_num:int):
    kp = 0.1
    kd = 0.01
    # y_des = 0
    y_real = model_data.body('frame_0-1').xpos[1]
    # ang_real = model_data.joint(f'Actuator{actuator_num}').qpos
    vel_real = model_data.joint(f'Actuator{actuator_num}').qvel
    error = np.cos(y_real)
    pd = kp * error
    print(f'{actuator_num}: {error}')
    return pd

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.type = 1
    viewer.cam.trackbodyid = 0
    viewer.cam.distance = 2

    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(2)

    while viewer.is_running():

        step_start = time.time()
        # print(data.body('frame_0-1').xpos)
        head_x.append(data.body('seg_1').xpos[0])
        head_y.append(data.body('seg_1').xpos[1])
        data.ctrl[:] = move_snake()
        # print(data.ctrl[:])
        mujoco.mj_step(model, data)
        viewer.sync()
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
        time.sleep(time_until_next_step)

plt.plot(head_x, head_y)
plt.title("Joint 1")
plt.ylabel("y, m")
plt.xlabel("x, m")
plt.grid(True)
plt.show()

