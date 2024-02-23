def plot_nematic_3D(self,Scalar = None, director = False, Flow = False, Plane = None):
        """
        Plots the nematic field in 3D using mayavi

        Input:
            scalar (numpy.narray, optional): the scalar field that we want to plot, S is ploted if None
            ax (Axes, optional): The axes to plot the field on. If not provided, a new subplot will be created.
            step (int, optional): The step size in the plot. Default is None.
            plane (list, optional): The plane that we want to plot the field on. Default is None.
            point (list, optional): The point that we want to plot the field on. Default is None.
            colormap (str, optional): The colormap to use for plotting the field. Default is None.
        
        Output:
            matplotlib.axes.Axes: The axes on which the nematic is plotted.
        """
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')

        S, n = self.calc_order_and_director()
        if Scalar is None:
            Scalar = S

        scal = mlab.pipeline.scalar_field(X,Y,Z,Scalar)

        mlab.pipeline.iso_surface(scal,contours = [Scalar.min() +0.5*Scalar.ptp()],opacity = 0.3)
        mlab.pipeline.iso_surface(scal, contours=[Scalar.max() - 0.5 * Scalar.ptp()],opacity=0.1)
        mlab.colorbar(orientation='vertical')
       # mlab.pipeline.scalar_cut_plane(scal, plane_orientation = 'z_axes')

        if director:
            vec = mlab.pipeline.vector_field(X,Y,Z,n[0],n[1],n[2])
            mlab.pipeline.vector_cut_plane(vec,mask_points =4,line_width =1, scale_factor = 1.0,
                                           plane_orientation= 'z_axes',mode = 'cylinder')

        if Flow:
            self.conf_u(self.Q)
            mlab.flow(X,Y,Z,self.u[0],self.u[1],self.u[2],seed_scale =1,seed_resolution=10, integration_direction ='both')