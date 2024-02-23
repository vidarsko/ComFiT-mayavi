
   def plot_field(self, field, **kwargs):
        """
        Plots the given (real) field.
        
        Input:
            field (array-like): The field to be plotted.
            **kwargs: Keyword arguments for the plot.
                See github.com/vidarsko/ComFiT/blob/main/docs/ClassBaseSystem.md 
                for a full list of keyword arguments.
        
        Output:
            matplotlib.axes.Axes: The axes containing the plot.
        """

        if field.dtype == bool:
            field = field.astype(float)

        if self.dim == 1:

            ax = kwargs.get('ax', plt.gca())

            ax.plot(self.x/self.a0, field)

            ax = self.plot_set_axis_properties(ax=ax, **kwargs) 
            
            return ax


        if self.dim == 2:

            ax = kwargs.get('ax', plt.gca())
            
            # Set the colormap
            colormap = kwargs.get('colormap', 'viridis')

            if colormap == 'bluewhitered':
                colormap = tool_colormap_bluewhitered()

            elif colormap == 'sunburst':
                colormap = tool_colormap_sunburst()
            else:
                colormap = plt.get_cmap(colormap)

            # Value limits symmetric
            vlim_symmetric = kwargs.get('vlim_symmetric', False)

            X, Y = np.meshgrid(self.x, self.y, indexing='ij')

            pcm = ax.pcolormesh(X / self.a0, Y / self.a0, field, shading='gouraud', cmap=colormap)

            xlim = [self.xmin, self.xmax-self.dx]
            ylim = [self.ymin, self.ymax-self.dy]

            limits_provided = False
            if 'xlim' in kwargs:
                xlim = kwargs['xlim']
                limits_provided = True
            else:
                if 'xmin' in kwargs:
                    xlim[0] = kwargs['xmin']
                    limits_provided = True
                
                if 'xmax' in kwargs:
                    xlim[1] = kwargs['xmax']
                    limits_provided = True

            if 'ylim' in kwargs:
                ylim = kwargs['ylim']
                limits_provided = True
            else:
                if 'ymin' in kwargs:
                    ylim[0] = kwargs['ymin']
                    limits_provided = True
                    
                if 'ymax' in kwargs:
                    ylim[1] = kwargs['ymax']
                    limits_provided = True

            # If explicit limits are provided, use them to change the vlim ranges
            if limits_provided:
                region_to_plot = np.zeros(self.dims).astype(bool)
                region_to_plot[(xlim[0] <= X)*(X <= xlim[1])*(ylim[0] <= Y)*(Y <= ylim[1])] = True
                vlim = [np.min(field[region_to_plot]), np.max(field[region_to_plot])]
                print(vlim)
            else:
                vlim = [np.min(field), np.max(field)]
            
            # Set the value limitses
            if 'vlim' in kwargs:
                vlim = kwargs['vlim']
            else:
                if 'vmin' in kwargs:
                    vlim[0] = kwargs['vmin']
                if 'vmax' in kwargs:
                    vlim[1] = kwargs['vmax']

            if vlim[1] - vlim[0] < 1e-10:
                vlim = [vlim[0]-0.05, vlim[1]+0.05]

            pcm.set_clim(vmin=vlim[0], vmax=vlim[1])

            if 'vlim_symmetric' in kwargs:
                vlim_symmetric = kwargs['vlim_symmetric']
                if vlim_symmetric:
                    cmax = abs(field).max()
                    cmin = -cmax
                    pcm.set_clim(vmin=cmin, vmax=cmax)

            colorbar = kwargs.get('colorbar', True)

            if colorbar:
                cbar = plt.colorbar(pcm, ax=ax)
                
            ax = self.plot_set_axis_properties(ax=ax, **kwargs)

            return ax

        elif self.dim == 3:

            field_min = np.min(field)
            field_max = np.max(field)

            plotting_lib = kwargs.get('plotting_lib', 'matplotlib')

            X, Y, Z = np.meshgrid(self.x/self.a0, self.y/self.a0, self.z/self.a0, indexing='ij')

            number_of_layers = kwargs.get('number_of_layers', 1)
            
            if 'clim' in kwargs:
                    clim = kwargs['clim']
                    cmin = clim[0]
                    cmax = clim[1]
            else:
                cmin = field_min
                cmax = field_max

            if 'layer_values' in kwargs:
                layer_values = np.concatenate([[-np.inf], kwargs['layer_values'], [np.inf]])
            else: 
                layer_values = np.linspace(cmin, cmax, number_of_layers + 2)

            if plotting_lib == 'mayavi':
                            scene = kwargs.get('scene', mlab.figure())

                            contour = mlab.contour3d(X, Y, Z, field, contours=layer_values.tolist(), opacity=0.5, colormap='viridis')
                            # axes = mlab.axes(xlabel='x/a0', ylabel='y/a0', zlabel='z/a0', figure=scene, 
                            #         nb_labels=5, ranges=(0, 5, 0, 5, -1, 1))

                            self.plot_set_scene_properties(**kwargs)

                            colorbar = kwargs.get('colorbar', True)
                            if colorbar:
                                cb = mlab.colorbar(object=contour, nb_labels=5)



                            mlab.view(-135,60)
            

            elif plotting_lib == 'matplotlib':

                ax = kwargs.get('ax', plt.gcf().add_subplot(111, projection='3d'))
                
                # print("Layer values:", layer_values)

                if 'colormap' in kwargs:
                    colormap = kwargs['colormap']
                    if colormap == 'bluewhitered':
                        colormap = tool_colormap_bluewhitered()

                    elif colormap == 'sunburst':
                        colormap = tool_colormap_sunburst()

                    else:
                        colormap = plt.get_cmap(colormap)
                else: 
                    colormap = plt.get_cmap('viridis')
                

                if field_min < layer_values[1] < field_max:
                    verts, faces, _, _ = marching_cubes(field, layer_values[1])
                    ax.plot_trisurf(self.xmin+verts[:, 0]*self.dx, self.ymin+verts[:, 1]*self.dy, faces, self.zmin+verts[:, 2]*self.dz, alpha=0.5,
                                color=colormap(layer_values[1] / cmax))

                for layer_value in layer_values[2:-1]:
                    if field_min < layer_value < field_max:
                        verts, faces, _, _ = marching_cubes(field, layer_value)
                        ax.plot_trisurf(self.xmin+verts[:, 0]*self.dx, self.ymin+verts[:, 1]*self.dy, faces, self.zmin+verts[:, 2]*self.dz, alpha=0.5,
                                    color=colormap(layer_value / cmax))

                ax.set_aspect('equal')

                if 'colorbar' in kwargs:
                    colorbar = kwargs['colorbar']
                else:
                    colorbar = True

                if colorbar:
                    sm = plt.cm.ScalarMappable(cmap=colormap)
                    sm.set_clim(cmin, cmax)
                    plt.colorbar(sm, ax=ax)

                ax.set_xlim3d(self.xmin, self.xmax-self.dx)
                ax.set_ylim3d(self.ymin, self.ymax-self.dy)
                ax.set_zlim3d(self.zmin, self.zmax-self.dz)
                ax.set_aspect('equal')
                ax.set_xlabel('$x/a_0$')
                ax.set_ylabel('$y/a_0$')
                ax.set_zlabel('$z/a_0$')


                return ax

def plot_shadows(self):
    if kwargs['plotting_lib'] == 'mayavi':
            if self.dim == 3:
                x = kwargs['x']
                y = kwargs['y']
                z = kwargs['z']
                faces = kwargs['faces']

                #Make xy-shadow:
                mesh = mlab.triangular_mesh(x, y, z*0+self.dx/2, faces, opacity=1, color=(0,0,0))

                #Make xz-shadow:
                mesh = mlab.triangular_mesh(x, y*0+self.ymax-self.dy/2, z, faces, opacity=1, color=(0,0,0))

                #Make yz-shadow:
                mesh = mlab.triangular_mesh(x*0+self.dx/2, y, z, faces, opacity=1, color=(0,0,0))
            else:
                raise Exception("Shadows are only implemented for 3D systems.")

def plot_set_scene_properties(self, **kwargs):
        """
        Sets the properties of the scene for a plot.
        """
        if self.dim == 3:
            
            Delta_x = (self.xmax - self.xmin)/5
            Delta_y = (self.ymax - self.ymin)/5
            Delta_z = (self.zmax - self.zmin)/5

            color = (0.764,0.922,0.969)
            opacity = 0.5

            for i in range(5):
                for j in range(5):
                    # Define the range for the grid
                    x_range =np.array([self.xmin+i*Delta_x+self.dx,self.xmin+(i+1)*Delta_x-self.dx])
                    y_range =np.array([self.ymin+j*Delta_y+self.dy,self.ymin+(j+1)*Delta_y-self.dy])
                    x, y = np.meshgrid(x_range, y_range)

                    # Height of the grid (z-coordinates)
                    z = np.zeros_like(x)+self.zmin

                    # Plot the grid as a flat surface
                    grid_surf = mlab.mesh(x, y, z, color=color,opacity=opacity)

            for i in range(5):
                for j in range(5):
                    # Define the range for the grid
                    x_range =np.array([self.xmin+i*Delta_x+self.dx,self.xmin+(i+1)*Delta_x-self.dx])
                    z_range =np.array([self.zmin+j*Delta_z+self.dz,self.zmin+(j+1)*Delta_z-self.dz])

                    x, z = np.meshgrid(x_range, z_range)

                    # Height of the grid (z-coordinates)
                    y = np.zeros_like(x) + self.ymax

                    # Plot the grid as a flat surface
                    grid_surf = mlab.mesh(x, y, z, color=color,opacity=opacity)

            for i in range(5):
                for j in range(5):
                    # Define the range for the grid
                    y_range =np.array([self.ymin+i*Delta_y+self.dy,self.ymin+(i+1)*Delta_y-self.dy])
                    z_range =np.array([self.zmin+j*Delta_z+self.dz,self.zmin+(j+1)*Delta_z-self.dz])

                    y, z = np.meshgrid(y_range, z_range)

                    # Height of the grid (z-coordinates)
                    x = np.zeros_like(y) + self.zmin

                    # Plot the grid as a flat surface
                    grid_surf = mlab.mesh(x, y, z, color=color,opacity=opacity)
            
            #Create axes:
            # Define the vertices of the cube
            vertices = np.array([self.xmin, self.ymin, self.zmin]) \
                            +np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])\
                                    *np.array([self.xmax-self.xmin, self.ymax-self.ymin, self.zmax-self.zmin])

            # Define the triangles that make up the six faces of the cube
            # Each face is made up of 2 triangles
            faces = np.array([[0, 1, 2], [0, 2, 3],  # Bottom
                                [4, 5, 6], [4, 6, 7],  # Top
                                [0, 1, 5], [0, 5, 4],  # Side
                                [2, 3, 7], [2, 7, 6],  # Opposite side
                                [0, 3, 7], [0, 7, 4],  # Front
                                [1, 2, 6], [1, 6, 5]]) # Back

            # Extract the x, y, z coordinates from vertices
            x, y, z = vertices.T

            # Use mlab.triangular_mesh to plot the triangles
            cube = mlab.triangular_mesh(x, y, z, faces, opacity=0)
            axes = mlab.axes(xlabel='x/a0', ylabel='y/a0', zlabel='z/a0', 
                    nb_labels=6)

            mlab.view(-45,60)
            mlab.gcf().scene.reset_zoom()

            
    def plot_angle_field(self, field, ax=None, colorbar=True):
        """
        Plot the angle field.

        Input:
            field (array-like): The angle field values.
            ax (matplotlib.axes.Axes, optional): The axes to plot the angle field on. If not provided, a new subplot will be created.
        
        Output:
            matplotlib.axes.Axes: The axes containing the plot.
        """

        if self.dim == 2:

            if ax is None:
                ax = plt.gca()

            X, Y = np.meshgrid(self.x, self.y, indexing='ij')

            custom_colormap = tool_colormap_angle()

            mesh = ax.pcolormesh(X, Y, field, shading='auto', cmap=custom_colormap, vmin=-np.pi, vmax=np.pi)
            if colorbar:
                cbar = plt.colorbar(mesh)  # To add a colorbar on the side
                cbar.set_ticks(np.array([-np.pi, -2 * np.pi / 3, -np.pi / 3, 0, np.pi / 3, 2 * np.pi / 3, np.pi]))
                cbar.set_ticklabels([r'$-\pi$', r'$-2\pi/3$', r'$-\pi/3$', r'$0$', r'$\pi/3$', r'$2\pi/3$', r'$\pi$'])
            # ax.title("Angle field")
            ax.set_xlabel('$x/a_0$')
            ax.set_ylabel('$y/a_0$')
            ax.set_aspect('equal')

            return ax

        elif self.dim == 3:

            if ax == None:
                plt.figure()
                ax = plt.gcf().add_subplot(111, projection='3d')

            X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')

            colormap = tool_colormap_angle()

            field_min = np.min(field)
            field_max = np.max(field)

            for angle in [-2 * np.pi / 3, -np.pi / 3, 0, np.pi / 3, 2 * np.pi / 3]:

                if field_min < angle < field_max:
                    field_to_plot = field.copy()
                    field_to_plot[field < angle - 1] = float('nan')
                    field_to_plot[field > angle + 1] = float('nan')

                    verts, faces, _, _ = marching_cubes(field_to_plot, angle)

                    ax.plot_trisurf(self.xmin+verts[:, 0]*self.dx, self.ymin+verts[:, 1]*self.dy, faces, self.zmin+verts[:, 2]*self.dz, alpha=0.5,
                                    color=colormap((angle + np.pi) / (2 * np.pi)))

            field = np.mod(field, 2 * np.pi)

            field_to_plot = field.copy()
            field_to_plot[field < np.pi - 1] = float('nan')
            field_to_plot[field > np.pi + 1] = float('nan')

            verts, faces, _, _ = marching_cubes(field_to_plot, np.pi)

            ax.plot_trisurf(self.xmin+verts[:, 0]*self.dx, self.ymin+verts[:, 1]*self.dy, faces, self.zmin+verts[:, 2]*self.dz, alpha=0.5,
                            color=colormap(0))

            ax.set_xlim3d(self.xmin, self.xmax-self.dx)
            ax.set_ylim3d(self.ymin, self.ymax-self.dy)
            ax.set_zlim3d(self.zmin, self.zmax-self.dz)
            ax.set_aspect('equal')

    

    def plot_fourier_field(self, field_f, ax=None):
        """
            Plot a Fourier field.

            Input:
                field_f (ndarray): The Fourier field to be plotted.
                ax (Axes3D, optional): The matplotlib 3D axis to be used for plotting. If not provided, a new axis will be created.

            Output:
                matplotlib.axes.Axes: The axes containing the plot.
            """
        field_f = np.fft.fftshift(field_f)

        if ax == None:
            ax = plt.gcf().add_subplot(111, projection='3d')

        if self.dim == 2:
            rho = np.abs(field_f)
            theta = np.angle(field_f)

            Kx, Ky = np.meshgrid(self.k[0], self.k[1], indexing='ij')

            Kx = np.fft.fftshift(Kx)
            Ky = np.fft.fftshift(Ky)

            custom_colormap = tool_colormap_angle()

            # Get the colors from a colormap (e.g., hsv, but you can choose any other)
            colors = plt.cm.hsv((theta + np.pi) / (2 * np.pi))  # Normalizing theta to [0, 1]
            surf = ax.plot_surface(Kx, Ky, rho, facecolors=colors, shade=True)

            return ax
            # mappable = plt.cm.ScalarMappable(cmap=custom_colormap)
            # mappable.set_array([])
            # mappable.set_clim(-np.pi, np.pi)
            # cbar = plt.colorbar(mappable, ax=ax)
            # cbar.set_ticks(np.array([-np.pi, -2 * np.pi / 3, -np.pi / 3, 0, np.pi / 3, 2 * np.pi / 3, np.pi]))
            # cbar.set_ticklabels([r'$-\pi$', r'$-2\pi/3$', r'$-\pi/3$', r'$0$', r'$\pi/3$', r'$2\pi/3$', r'$\pi$'])

            # plt.title("Angle field")
            # plt.xlabel("X-axis")
            # plt.ylabel("Y-axis")

    def plot_complex_field(self, complex_field, **kwargs):
        """
        Plot a complex field.

        ax=None, plot_method=None, colorbar=False

        Input:
            complex_field (numpy.ndarray): The complex field to plot.
            ax (matplotlib.axes.Axes, optional): The matplotlib axes on which to plot the field.
                If not provided, a new 3D axes will be created.
        
        Output:
            matplotlib.axes.Axes: The axes containing the plot.
                
        Raises:
            Exception: If the dimension of the field is not 2.
        """



        if self.dim == 2:

            plot_method = kwargs.get('plot_method', 'phase_angle')
            plotting_lib = kwargs.get('plotting_lib', 'matplotlib')

            if plot_method == '3Dsurface':

                if plotting_lib == 'matplotlib':

                    ax = kwargs.get('ax', None)

                    if ax == None:
                        plt.clf()
                        ax = plt.gcf().add_subplot(111, projection='3d')

                    X, Y = np.meshgrid(self.x, self.y, indexing='ij')

                    custom_colormap = tool_colormap_angle()
                    rho = np.abs(complex_field)
                    theta = np.angle(complex_field)
                    # Get the colors from a colormap (e.g., hsv, but you can choose any other)
                    colors = plt.cm.hsv((theta + np.pi) / (2 * np.pi))  # Normalizing theta to [0, 1]

                    surf = ax.plot_surface(X, Y, rho, facecolors=colors)
                    
                    ax.set_xlabel("$x/a_0$")
                    ax.set_ylabel("$y/a_0$")
                    return ax
                
                elif plotting_lib == 'mayavi':
                    X, Y = np.meshgrid(self.x, self.y, indexing='ij')
                    rho = np.abs(complex_field)
                    theta = np.angle(complex_field)

                    # Normalize theta for colormap application
                    theta_faces_normalized = (theta + np.pi) / (2 * np.pi)

                    # Plotting the surface
                    surf = mlab.mesh(X/self.a0, Y/self.a0, 20*rho, scalars=theta_faces_normalized, colormap='hsv')

                    axes = mlab.axes(xlabel='x/a0', ylabel='y/a0',  
                        nb_labels=5)
                    mlab.view(-135,60,300)
                    

                    # Setting the color range to match the normalized theta range
                    # surf.module_manager.scalar_lut_manager.data_range = np.array([0, 1])

                    # Labels (Mayavi manages labels and axes differently than Matplotlib)
                    # mlab.xlabel("$x/a_0$")
                    # mlab.ylabel("$y/a_0$")


            elif plot_method == 'phase_angle':

                ax = kwargs.get('ax', None)

                if ax == None:
                    plt.clf()
                    ax = plt.gca()

                X, Y = np.meshgrid(self.x, self.y, indexing='ij')

                rho = np.abs(complex_field)
                theta = np.angle(complex_field)

                rho_normalized = rho / np.max(rho)
                custom_colormap = tool_colormap_angle()


                # Create a new colormap for magnitudeW
                # Starting from white (for zero magnitude) to the full color of the phase
                # custom_colormap_mag = mcolors.LinearSegmentedColormap.from_list(
                #     'MagnitudeColorMap',
                #     [(1, 1, 1, 0), custom_colormap_phase(1.0)],
                #     N=256
                # )

                # Calculate colors based on magnitude and phase
                #colors = custom_colormap_phase(theta)
                #colors[..., 3] = rho_normalized  # Set the alpha channel according to the magnitude

                mesh = ax.pcolormesh(X, Y, theta, shading='auto', cmap=custom_colormap, vmin=-np.pi, vmax=np.pi)
                mesh.set_alpha(rho_normalized)
                
                #mesh.set_array(None)  # Avoids warning
                #mesh.set_edgecolor('face')
                #mesh.set_facecolor(colors)  # Use the calculated colors

                colorbar = kwargs.get('colorbar', True)

                if colorbar:
                    mappable = plt.cm.ScalarMappable(cmap=custom_colormap)
                    mappable.set_array([])
                    mappable.set_clim(-np.pi, np.pi)
                    cbar = plt.colorbar(mappable, ax=ax)
                    cbar.set_ticks(np.array([-np.pi, -2 * np.pi / 3, -np.pi / 3, 0, np.pi / 3, 2 * np.pi / 3, np.pi]))
                    cbar.set_ticklabels([r'$-\pi$', r'$-2\pi/3$', r'$-\pi/3$', r'$0$', r'$\pi/3$', r'$2\pi/3$', r'$\pi$'])

                ax.set_xlabel("$x/a_0$")
                ax.set_ylabel("$y/a_0$")
                ax.set_aspect('equal')
            
                return ax

        elif self.dim == 3:

            plot_method = kwargs.get('plot_method', 'phase_blob')
            plotting_lib = kwargs.get('plotting_lib', 'matplotlib')
            

            X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')

            rho = np.abs(complex_field)
            rho_normalized = rho / np.max(rho)
            theta = np.angle(complex_field)

            colormap = tool_colormap_angle()

            if plotting_lib == 'matplotlib':
                ax = kwargs.get('ax', None)

                if ax == None:
                    plt.clf()
                    ax = plt.gcf().add_subplot(111, projection='3d')
        
            if plot_method == 'phase_angle':
                

                for angle in [-2 * np.pi / 3, -np.pi / 3, 0, np.pi / 3, 2 * np.pi / 3]:
                    field_to_plot = theta.copy()
                    field_to_plot[theta < angle - 1] = float('nan')
                    field_to_plot[theta > angle + 1] = float('nan')
                    field_to_plot[rho_normalized < 0.01] = float('nan')

                    if np.nanmin(field_to_plot) < angle < np.nanmax(field_to_plot):

                        verts, faces, _, _ = marching_cubes(field_to_plot, angle)

                        ax.plot_trisurf(self.xmin+verts[:, 0]*self.dx, self.ymin+verts[:, 1]*self.dy, faces, self.zmin+verts[:, 2]*self.dz, alpha=0.5,
                                        color=colormap((angle + np.pi) / (2 * np.pi)))

                theta = np.mod(theta, 2 * np.pi)

                field_to_plot = theta.copy()
                field_to_plot[theta < np.pi - 1] = float('nan')
                field_to_plot[theta > np.pi + 1] = float('nan')
                field_to_plot[rho_normalized < 0.01] = float('nan')

                if np.nanmin(field_to_plot) < np.pi < np.nanmax(field_to_plot):

                    verts, faces, _, _ = marching_cubes(field_to_plot, np.pi)

                    ax.plot_trisurf(self.xmin+verts[:, 0]*self.dx, self.ymin+verts[:, 1]*self.dy, faces, self.zmin+verts[:, 2]*self.dz, alpha=0.5,
                                color=colormap(0))
            
            elif plot_method == 'phase_blob':
                if np.nanmin(rho_normalized)<0.5<np.nanmax(rho_normalized):
                    verts, faces, _, _ = marching_cubes(rho_normalized, 0.5)

                    # Calculate the centroids of each triangle
                    centroids = np.mean(verts[faces], axis=1)

                    # Assuming theta is defined on the same grid as rho
                    x, y, z = np.mgrid[0:rho_normalized.shape[0], 0:rho_normalized.shape[1], 0:rho_normalized.shape[2]]
                    x = self.xmin+x*self.dx
                    y = self.ymin+y*self.dy
                    z = self.zmin+z*self.dz

                    # Flatten the grid for interpolation
                    points = np.c_[x.ravel(), y.ravel(), z.ravel()]
                    theta_values = theta.ravel()

                    # Interpolate theta at the vertices positions
                    theta_faces = sp.interpolate.griddata(points, theta_values, centroids, method='nearest')

                    # Normalize theta values for color mapping
                    theta_faces_normalized = (theta_faces + np.pi) / (2*np.pi)

                    # Map normalized theta values to colors
                    colors = colormap(theta_faces_normalized)

                if plotting_lib == 'mayavi':

                    fig = kwargs.get('fig', 
                            mlab.gcf() if mlab.get_engine() else mlab.figure(bgcolor=(0.5, 0.5, 0.5)))
                    
                    # TODO: It might be faster to update the data sources for plotting with mayavi (Vidar 15.02.24)
                    hold = kwargs.get('hold', False)
                    if not hold:
                        mlab.clf()

                    if np.nanmin(rho_normalized)<0.5<np.nanmax(rho_normalized):
                    # print("theta_faces_normalized:", theta_faces_normalized)
                        theta_faces_normalized = np.mod(theta_faces_normalized+0.5,1)

                        vertex_values = np.zeros(verts.shape[0])
                        for i, face in enumerate(faces):
                            for vert in face:
                                vertex_values[vert] = theta_faces_normalized[i]
                        
                        # Now, create the mesh using mlab.triangular_mesh
                        x, y, z = verts.T  # Transpose verts to get separate arrays
                        
                        mesh = mlab.triangular_mesh(x, y, z, faces, representation='surface', opacity=1,
                                                    scalars=vertex_values, colormap='hsv')

                        self.plot_shadows(x=x,y=y,z=z,faces=faces,plotting_lib=plotting_lib)

                    self.plot_set_scene_properties(**kwargs)

                    # Make sure the scene is rendered
                    QApplication.processEvents()
                    return fig
                
                elif plotting_lib == 'matplotlib':
                    # print("Colors shape:", colors.shape)
                    # print(colors)

                    ax.plot_trisurf(self.xmin+verts[:, 0]*self.dx, 
                                    self.ymin+verts[:, 1]*self.dy, 
                                    faces, 
                                    self.zmin+verts[:, 2]*self.dz, 
                                    facecolor=colors, antialiased=False)

                    ax.set_xlim3d(self.xmin, self.xmax-self.dx)
                    ax.set_ylim3d(self.ymin, self.ymax-self.dy)
                    ax.set_zlim3d(self.zmin, self.zmax-self.dz)

                    ax.set_aspect('equal')

                    ax.set_xlabel("$x/a_0$")
                    ax.set_ylabel("$y/a_0$")
                    ax.set_zlabel("$z/a_0$")
                    ax.set_aspect('equal')

                    return ax

        else:
            raise Exception("This plotting function not yet configured for other dimension")

    def plot_field_in_plane(self, field, normal_vector=[0,1,0], position=None, ax=None,
                         colorbar=True, colormap='viridis', clim=None, plotting_lib='matplotlib',
                         **kwargs):
        """
        Plots the field in a plane perpendicular to the given normal vector using
        scipy.interpolate.griddata and plt.plot_trisurf.

        Input:
            field (array-like): The field to be plotted.
            normal_vector (array-like, optional): The normal vector of the plane. Default is [0,1,0].
            position (array-like, optional): The position of the plane. Default is the middle of the system.
            ax (Axes, optional): The axes object to plot on. If None, a new figure and axes will be created.
            colorbar (bool, optional): Whether to include a colorbar in the plot. Default is True.
            colormap (str, optional): The colormap to use for the plot. 
        
        Output:
            matplotlib.axes.Axes: The axes containing the plot.
        """

        if self.dim != 3:
            raise Exception("This plotting function not yet configured for other dimensions")

        if position is None:
            position = self.rmid

        if ax is None:
            plt.clf()
            ax = plt.gcf().add_subplot(111, projection='3d')

        if colormap == 'angle':
            colormap = tool_colormap_angle()
        elif colormap == 'bluewhitered':
            colormap = tool_colormap_bluewhitered()
        else:
            colormap = plt.get_cmap(colormap)


        normal_vector = np.array(normal_vector)/np.linalg.norm(normal_vector)
        height_above_plane = (self.x-position[0])*normal_vector[0] + (self.y-position[1])*normal_vector[1] + (self.z-position[2])*normal_vector[2]

        verts, faces, _, _ = marching_cubes(height_above_plane, 0)

        # Calculate the centroids of each triangle
        centroids = np.mean(verts[faces], axis=1)

        # Assuming field is defined on the same grid as height_above_plane
        x, y, z = np.mgrid[0:height_above_plane.shape[0], 0:height_above_plane.shape[1], 0:height_above_plane.shape[2]]

        # Flatten the grid for interpolation
        points = np.c_[x.ravel(), y.ravel(), z.ravel()]
        field_values = field.ravel()

        # Interpolate field at the vertices positions
        field_verts = sp.interpolate.griddata(points, field_values, centroids, method='nearest')

        # Normalize field values for color mapping
        field_normalized = (field_verts - np.min(field_verts)) / (np.max(field_verts) - np.min(field_verts))

        # Map normalized field values to colors
        colors = colormap(field_normalized)


        if plotting_lib == 'matplotlib':

            ax.plot_trisurf(self.xmin+verts[:, 0]*self.dx,
                            self.ymin+verts[:, 1]*self.dy,
                            faces,
                            self.zmin+verts[:, 2]*self.dz,
                            facecolor=colors, antialiased=False)

            if colorbar:
                sm = plt.cm.ScalarMappable(cmap=colormap)    
                cbar = plt.colorbar(sm, ax=ax)

            ax.set_xlim3d(self.xmin, self.xmax-self.dx)
            ax.set_ylim3d(self.ymin, self.ymax-self.dy)
            ax.set_zlim3d(self.zmin, self.zmax-self.dz)

            ax.set_aspect('equal')

            ax.set_xlabel("$x/a_0$")
            ax.set_ylabel("$y/a_0$")
            ax.set_zlabel("$z/a_0$")
            ax.set_aspect('equal')

            return ax
        
        elif plotting_lib == 'mayavi':
            fig = kwargs.get('fig', 
                            mlab.gcf() if mlab.get_engine() else mlab.figure(bgcolor=(0.5, 0.5, 0.5)))
                    
            # TODO: It might be faster to update the data sources for plotting with mayavi (Vidar 15.02.24)
            hold = kwargs.get('hold', False)
            if not hold:
                mlab.clf()

            vertex_values = np.zeros(verts.shape[0])
            for i, face in enumerate(faces):
                for vert in face:
                    vertex_values[vert] = field_normalized[i]
            
            # Now, create the mesh using mlab.triangular_mesh
            x, y, z = verts.T  # Transpose verts to get separate arrays
            x = self.xmin+x*self.dx
            y = self.ymin+y*self.dy
            z = self.zmin+z*self.dz
            
            mesh = mlab.triangular_mesh(x, y, z, faces, representation='surface', opacity=1,
                                        scalars=vertex_values, colormap='viridis')

            self.plot_set_scene_properties(**kwargs)

            colorbar = kwargs.get('colorbar', True)
            if colorbar:
                cb = mlab.colorbar(object=mesh, nb_labels=5)

            # Make sure the scene is rendered
            QApplication.processEvents()
            return fig

    def plot_angle_field_in_plane(self, angle_field, colorbar=True):
        """
        Plots the angle field in a plane.

        Input:
            angle_field (numpy.ndarray): The angle field to be plotted.
            colorbar (bool, optional): Whether to include a colorbar. Defaults to True.
        
        Output:
            matplotlib.axes.Axes: The axes containing the plot.
        """
        self.plot_field_in_plane(angle_field, colorbar=False)

        if colorbar:
            sm = plt.cm.ScalarMappable(cmap=tool_colormap_angle())
            sm.set_clim(-np.pi, np.pi)
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_ticks(np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]))
            cbar.set_ticklabels([r'$-\pi$', r'$-2\pi/3$', r'$-\pi/3$', r'$0$', r'$\pi/3$', r'$2\pi/3$', r'$\pi$'])

    
        

    def plot_vector_field(self, vector_field, ax=None, step=None):
        """
        Plots a vector field on a 2D grid.

        Input:
        vector_field (tuple): Tuple containing the x and y components of the vector field.
        ax (matplotlib.axes.Axes, optional): The axes on which to plot the vector field. If not provided, a new subplot will be created.

        Output:
        matplotlib.axes.Axes: The axes containing the plot.
        """

        if self.dim == 2:

            if ax == None:
                ax = plt.gcf().add_subplot(111)

            if step == None:
                step = 5

            X, Y = np.meshgrid(self.x, self.y, indexing='ij')

            X_plot = X[::step, ::step]
            Y_plot = Y[::step, ::step]
            U_plot = vector_field[0][::step, ::step]
            V_plot = vector_field[1][::step, ::step]

            max_vector = np.max(np.sqrt(U_plot ** 2 + V_plot ** 2))
            print(max_vector)

            ax.quiver(X_plot, Y_plot, U_plot, V_plot, scale=25 * max_vector / step)

            ax.set_xlabel('$x/a_0$')
            ax.set_ylabel('$y/a_0$')
            ax.set_aspect('equal')
            ax.set_xlim([0, self.xmax-self.dx])
            ax.set_ylim([0, self.ymax-self.dy])

        elif self.dim == 3:

            if ax == None:
                ax = plt.gcf().add_subplot(111, projection='3d')

            if step is None:
                step = 2

            X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')

            X_plot = X[::step, ::step, ::step]
            Y_plot = Y[::step, ::step, ::step]
            Z_plot = Z[::step, ::step, ::step]
            U_plot = vector_field[0][::step, ::step, ::step]
            V_plot = vector_field[1][::step, ::step, ::step]
            W_plot = vector_field[2][::step, ::step, ::step]

            max_vector = np.max(np.sqrt(U_plot ** 2 + V_plot ** 2 + W_plot ** 2))

            U_plot = U_plot / max_vector
            V_plot = V_plot / max_vector
            W_plot = W_plot / max_vector

            ax.quiver(X_plot, Y_plot, Z_plot, U_plot, V_plot, W_plot, arrow_length_ratio=0.6)

            ax.set_xlabel('$x/a_0$')
            ax.set_ylabel('$y/a_0$')
            ax.set_zlabel('$z/a_0$')
            ax.set_aspect('equal')
            ax.set_xlim([0, self.xmax-self.dx])
            ax.set_ylim([0, self.ymax-self.dy])
            ax.set_zlim([0, self.zmax-self.dz])

        return ax




