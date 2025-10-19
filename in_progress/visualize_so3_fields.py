"""
Visualization utilities for SO(3) vector fields and 3D sphere attractors.

This module provides various ways to visualize 3D vector fields, particularly
SO(3) rotation generators and learned dynamics on the sphere.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sphere_dynamics import create_so3_generators, create_combined_so3_vector_field


def plot_3d_vector_field_on_sphere(vector_field_fn, n_points=15, scale=0.2, 
                                   title="3D Vector Field", ax=None, color='blue',
                                   show_sphere=True, alpha=0.6):
    """
    Plot a 3D vector field on the surface of a sphere using arrows.
    
    Args:
        vector_field_fn: Function that takes (N, 3) points and returns (N, 3) vectors
        n_points: Number of points along each spherical coordinate
        scale: Scale factor for arrow lengths
        title: Plot title
        ax: Matplotlib 3D axis (creates new if None)
        color: Arrow color
        show_sphere: Whether to show sphere wireframe
        alpha: Transparency for arrows
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Create points on unit sphere using spherical coordinates
    theta = np.linspace(0, 2*np.pi, n_points)  # azimuthal angle
    phi = np.linspace(0, np.pi, n_points//2)   # polar angle (avoid poles)
    
    points = []
    for t in theta:
        for p in phi:
            x = np.sin(p) * np.cos(t)
            y = np.sin(p) * np.sin(t)
            z = np.cos(p)
            points.append([x, y, z])
    
    points = torch.tensor(points, dtype=torch.float32)
    
    # Evaluate vector field
    with torch.no_grad():
        vectors = vector_field_fn(points)
    
    # Plot arrows
    ax.quiver(points[:, 0].numpy(), points[:, 1].numpy(), points[:, 2].numpy(),
              vectors[:, 0].numpy(), vectors[:, 1].numpy(), vectors[:, 2].numpy(),
              length=scale, color=color, alpha=alpha, arrow_length_ratio=0.1)
    
    # Show sphere wireframe
    if show_sphere:
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.1, color='gray')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Equal aspect ratio
    max_range = 1.5
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])


def plot_so3_generators_grid(figsize=(15, 5)):
    """
    Plot all three SO(3) generators in a row for comparison.
    """
    generators = create_so3_generators()
    generator_names = ['X-rotation', 'Y-rotation', 'Z-rotation']
    colors = ['red', 'green', 'blue']
    
    fig = plt.figure(figsize=figsize)
    
    for i, (gen, name, color) in enumerate(zip(generators, generator_names, colors)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        plot_3d_vector_field_on_sphere(
            gen, ax=ax, title=f"SO(3) {name}", 
            color=color, scale=0.3, n_points=12
        )
    
    plt.tight_layout()
    plt.show()


def plot_2d_projections_so3(vector_field_fn, title="SO(3) Field Projections", figsize=(12, 4)):
    """
    Show 2D projections (XY, XZ, YZ) of a 3D vector field.
    Useful for understanding 3D structure in familiar 2D views.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Create grid of points on sphere
    n_points = 100
    theta = np.linspace(0, 2*np.pi, int(np.sqrt(n_points)))
    phi = np.linspace(0.1, np.pi-0.1, int(np.sqrt(n_points)))  # Avoid poles
    
    points = []
    for t in theta:
        for p in phi:
            x = np.sin(p) * np.cos(t)
            y = np.sin(p) * np.sin(t) 
            z = np.cos(p)
            points.append([x, y, z])
    
    points = torch.tensor(points, dtype=torch.float32)
    
    with torch.no_grad():
        vectors = vector_field_fn(points)
    
    # XY projection
    axes[0].quiver(points[:, 0].numpy(), points[:, 1].numpy(),
                   vectors[:, 0].numpy(), vectors[:, 1].numpy(),
                   scale=10, alpha=0.7, color='blue')
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', alpha=0.3)
    axes[0].add_patch(circle)
    axes[0].set_title('XY Projection')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-1.5, 1.5)
    axes[0].set_ylim(-1.5, 1.5)
    
    # XZ projection
    axes[1].quiver(points[:, 0].numpy(), points[:, 2].numpy(),
                   vectors[:, 0].numpy(), vectors[:, 2].numpy(),
                   scale=10, alpha=0.7, color='red')
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', alpha=0.3)
    axes[1].add_patch(circle)
    axes[1].set_title('XZ Projection')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-1.5, 1.5)
    axes[1].set_ylim(-1.5, 1.5)
    
    # YZ projection
    axes[2].quiver(points[:, 1].numpy(), points[:, 2].numpy(),
                   vectors[:, 1].numpy(), vectors[:, 2].numpy(),
                   scale=10, alpha=0.7, color='green')
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', alpha=0.3)
    axes[2].add_patch(circle)
    axes[2].set_title('YZ Projection')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(-1.5, 1.5)
    axes[2].set_ylim(-1.5, 1.5)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_streamlines_on_sphere(vector_field_fn, n_streamlines=8, max_length=2*np.pi, 
                              title="Streamlines on Sphere", figsize=(10, 8)):
    """
    Plot streamlines (integral curves) of the vector field on the sphere surface.
    This shows the flow patterns more clearly than just arrows.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample starting points on sphere
    np.random.seed(42)  # For reproducibility
    start_points = []
    for _ in range(n_streamlines):
        # Random point on unit sphere
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0.2, np.pi-0.2)  # Avoid poles
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        start_points.append([x, y, z])
    
    start_points = torch.tensor(start_points, dtype=torch.float32)
    
    # Integrate streamlines using simple Euler method
    dt = 0.05
    max_steps = int(max_length / dt)
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_streamlines))
    
    for i, start_point in enumerate(start_points):
        trajectory = [start_point.clone()]
        current_point = start_point.clone()
        
        for step in range(max_steps):
            with torch.no_grad():
                vector = vector_field_fn(current_point.unsqueeze(0)).squeeze(0)
            
            # Project vector to be tangent to sphere (for sphere surface flow)
            current_point_normalized = current_point / torch.norm(current_point)
            tangent_vector = vector - torch.dot(vector, current_point_normalized) * current_point_normalized
            
            # Take step
            next_point = current_point + dt * tangent_vector
            # Project back to sphere surface
            next_point = next_point / torch.norm(next_point)
            
            trajectory.append(next_point)
            current_point = next_point
            
            # Stop if we've made a complete loop (roughly)
            if step > 10 and torch.norm(current_point - start_point) < 0.1:
                break
        
        trajectory = torch.stack(trajectory)
        ax.plot(trajectory[:, 0].numpy(), 
               trajectory[:, 1].numpy(), 
               trajectory[:, 2].numpy(), 
               color=colors[i], linewidth=2, alpha=0.8)
        
        # Mark starting point
        ax.scatter(start_point[0].item(), start_point[1].item(), start_point[2].item(),
                  color=colors[i], s=50, alpha=1.0)
    
    # Show sphere wireframe
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.1, color='gray')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Equal aspect ratio
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    
    plt.show()


def visualize_so3_vs_learned_dynamics(learned_dynamics_fn, so3_field_fn, 
                                     figsize=(20, 10)):
    """
    Compare learned dynamics with SO(3) target field using multiple visualizations.
    """
    fig = plt.figure(figsize=figsize)
    
    # Row 1: 3D quiver plots
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    plot_3d_vector_field_on_sphere(learned_dynamics_fn, ax=ax1, 
                                  title="Learned Dynamics", color='red', n_points=10)
    
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    plot_3d_vector_field_on_sphere(so3_field_fn, ax=ax2,
                                  title="SO(3) Target", color='blue', n_points=10)
    
    # Row 1: Streamlines
    ax3 = fig.add_subplot(2, 4, 3, projection='3d')
    plot_streamlines_on_sphere_ax(learned_dynamics_fn, ax3, title="Learned Flow", n_streamlines=6)
    
    ax4 = fig.add_subplot(2, 4, 4, projection='3d')
    plot_streamlines_on_sphere_ax(so3_field_fn, ax4, title="SO(3) Flow", n_streamlines=6)
    
    # Row 2: 2D projections (learned)
    ax5 = fig.add_subplot(2, 4, 5)
    plot_2d_projection_single(learned_dynamics_fn, ax5, 'XY', title="Learned XY")
    
    ax6 = fig.add_subplot(2, 4, 6)
    plot_2d_projection_single(learned_dynamics_fn, ax6, 'XZ', title="Learned XZ")
    
    # Row 2: 2D projections (target)
    ax7 = fig.add_subplot(2, 4, 7)
    plot_2d_projection_single(so3_field_fn, ax7, 'XY', title="Target XY")
    
    ax8 = fig.add_subplot(2, 4, 8)
    plot_2d_projection_single(so3_field_fn, ax8, 'XZ', title="Target XZ")
    
    plt.tight_layout()
    plt.show()


def plot_streamlines_on_sphere_ax(vector_field_fn, ax, n_streamlines=6, title="Streamlines"):
    """Helper function for streamlines on existing axis."""
    # Sample starting points
    np.random.seed(42)
    start_points = []
    for _ in range(n_streamlines):
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0.3, np.pi-0.3)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        start_points.append([x, y, z])
    
    start_points = torch.tensor(start_points, dtype=torch.float32)
    dt = 0.1
    max_steps = 50
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_streamlines))
    
    for i, start_point in enumerate(start_points):
        trajectory = [start_point.clone()]
        current_point = start_point.clone()
        
        for step in range(max_steps):
            with torch.no_grad():
                vector = vector_field_fn(current_point.unsqueeze(0)).squeeze(0)
            
            # Project to sphere surface
            current_point_normalized = current_point / torch.norm(current_point)
            tangent_vector = vector - torch.dot(vector, current_point_normalized) * current_point_normalized
            
            next_point = current_point + dt * tangent_vector
            next_point = next_point / torch.norm(next_point)
            
            trajectory.append(next_point)
            current_point = next_point
        
        trajectory = torch.stack(trajectory)
        ax.plot(trajectory[:, 0].numpy(), trajectory[:, 1].numpy(), trajectory[:, 2].numpy(), 
               color=colors[i], linewidth=2, alpha=0.8)
    
    # Sphere wireframe
    u = np.linspace(0, 2 * np.pi, 15)
    v = np.linspace(0, np.pi, 15)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.1, color='gray')
    
    ax.set_title(title)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])


def plot_2d_projection_single(vector_field_fn, ax, projection='XY', title="Projection"):
    """Helper function for single 2D projection."""
    # Create points on sphere
    n_points = 80
    theta = np.linspace(0, 2*np.pi, int(np.sqrt(n_points)))
    phi = np.linspace(0.2, np.pi-0.2, int(np.sqrt(n_points)))
    
    points = []
    for t in theta:
        for p in phi:
            x = np.sin(p) * np.cos(t)
            y = np.sin(p) * np.sin(t)
            z = np.cos(p)
            points.append([x, y, z])
    
    points = torch.tensor(points, dtype=torch.float32)
    
    with torch.no_grad():
        vectors = vector_field_fn(points)
    
    if projection == 'XY':
        ax.quiver(points[:, 0].numpy(), points[:, 1].numpy(),
                 vectors[:, 0].numpy(), vectors[:, 1].numpy(),
                 scale=15, alpha=0.7)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    elif projection == 'XZ':
        ax.quiver(points[:, 0].numpy(), points[:, 2].numpy(),
                 vectors[:, 0].numpy(), vectors[:, 2].numpy(),
                 scale=15, alpha=0.7)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
    elif projection == 'YZ':
        ax.quiver(points[:, 1].numpy(), points[:, 2].numpy(),
                 vectors[:, 1].numpy(), vectors[:, 2].numpy(),
                 scale=15, alpha=0.7)
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
    
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', alpha=0.3)
    ax.add_patch(circle)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)


if __name__ == "__main__":
    print("🌐 SO(3) Vector Field Visualization Examples")
    
    # Example 1: Show all SO(3) generators
    print("\n1. Plotting all SO(3) generators...")
    plot_so3_generators_grid()
    
    # Example 2: Show 2D projections of z-rotation
    print("\n2. 2D projections of Z-rotation generator...")
    z_rotation = create_so3_generators()[2]
    plot_2d_projections_so3(z_rotation, "Z-rotation SO(3) Generator")
    
    # Example 3: Streamlines of combined SO(3) field
    print("\n3. Streamlines of combined SO(3) field...")
    combined_field = create_combined_so3_vector_field(torch.tensor([0.5, 0.5, 1.0]))
    plot_streamlines_on_sphere(combined_field, title="Combined SO(3) Field Streamlines")
    
    print("✅ SO(3) visualization examples complete!")
    print("\nVisualization methods available:")
    print("- plot_3d_vector_field_on_sphere(): 3D quiver plot")  
    print("- plot_2d_projections_so3(): XY, XZ, YZ projections")
    print("- plot_streamlines_on_sphere(): Integral curves on sphere")
    print("- plot_so3_generators_grid(): All three generators")
    print("- visualize_so3_vs_learned_dynamics(): Compare learned vs target")