import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def visualize_npy(file_path="/home/theo/code/hand-retargeting-sam3d/data/sam3d_outputs/dexsuite_joints1.npy"):
    print(f"Loading {file_path}...")
    try:
        raw_data = np.load(file_path)
        print(f"Data successfully loaded! Shape: {raw_data.shape}")
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Make sure you are in the right folder.")
        return

    if len(raw_data.shape) == 4:
        print("Detected 2 hands in the dataset. Extracting Right Hand (Index 0)...")
        data = raw_data[:, 0, :, :] 
    else:
        data = raw_data

    num_frames = data.shape[0]
    num_joints = data.shape[1]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)

    ax.set_xlim([np.min(data[:, :, 0]), np.max(data[:, :, 0])])
    ax.set_ylim([np.min(data[:, :, 1]), np.max(data[:, :, 1])])
    ax.set_zlim([np.min(data[:, :, 2]), np.max(data[:, :, 2])])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"3D Kinematics - Frame 0")

    scatters = ax.scatter(data[0, :, 0], data[0, :, 1], data[0, :, 2], c='blue', s=20, depthshade=True)
    
    texts = []
    for i in range(num_joints):
        txt = ax.text(data[0, i, 0], data[0, i, 1], data[0, i, 2], str(i), size=8, zorder=1, color='k')
        texts.append(txt)

    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    frame_slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=num_frames - 1,
        valinit=0,
        valstep=1
    )

    def update(val):
        frame_idx = int(frame_slider.val)
        
        scatters._offsets3d = (data[frame_idx, :, 0], data[frame_idx, :, 1], data[frame_idx, :, 2])
        
        for i in range(num_joints):
            texts[i].set_position_3d((data[frame_idx, i, 0], data[frame_idx, i, 1], data[frame_idx, i, 2]))
            
        ax.set_title(f"3D Kinematics - Frame {frame_idx}")
        fig.canvas.draw_idle()

    frame_slider.on_changed(update)

    ax.view_init(elev=-90, azim=-90)

    print("Opening visualizer window... (Close the window to exit the script)")
    plt.show()

if __name__ == "__main__":
    file_path="/home/theo/code/hand-retargeting-sam3d/data/sam3d_outputs/dexsuite_joints1.npy"
    visualize_npy(file_path)