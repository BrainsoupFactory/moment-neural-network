# analyzing corr

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from mnn.mnn_core.mnn_pytorch import *
import sys
from scipy.spatial import ConvexHull

#exp_id = 
#exp_id = '2024_mar_20_neg_corr'
exp_id = '2024_mar_21_longer_T'

if exp_id == '2024_mar_19':
    num_files = 20
    xlim = [0,1.0]
    ylim = [-0.1,1.0]
elif exp_id in ['2024_mar_20_neg_corr', '2024_mar_21_longer_T']:
    num_files = 39
    xlim = [-1,1]
    ylim = [-1,1]

path =  './projects/pre2023/runs/{}/'.format( exp_id )

dat = np.load(path+'aggregated_data.npz')

mnn_corr_5d = dat['mnn_corr_5d']
snn_corr_5d = dat['snn_corr_5d']
mean_grid = dat['mean_grid']
std_grid = dat['std_grid']
corr_array = dat['corr_array']
corr_err_L2 = dat['corr_err_L2']
#print(dat)
#input_mean = dat['input_mean']
#input_std = dat['input_std']


print(corr_array)

if len(sys.argv)==1: # no additional inputs
    is_plot = [True]*10 #plot everything
else:
    is_plot = [False]*10
    is_plot[ int(sys.argv[1]) ] = True

def color_picker(slope):
    if slope <0:
        if np.abs(slope) < 0.25:
            color = ['#f0bfd5','#7c1d48'] #mean dominant
        else:
            color = ['#f8f09a','#8f840a'] #fluctuation dominant
    else:
        eff_drive = slope
        if eff_drive < 0.25:
            color = ['#e3d9e1','#584153'] #subthreshold
        elif eff_drive < 1.25:
            color = ['#9ec2e2','#234f76'] #balanced
        else:
            color = ['#f8f09a','#8f840a'] #fluctuation dominant
    return color

if is_plot[0]:
    print('1. Plotting diagonal slice...')
    plt.figure(figsize=(8,7.5))
    for i in range(mean_grid.shape[1]):
        plt.subplot(11,10, int(i+1))
        
        # color code line with regime #grab color scheme from moment_nn
        # solid/dashed to code mnn/snn
        L = 1/20
        slope = (np.sqrt(L)*std_grid[0,i])/(1.0-mean_grid[0,i])
        color = color_picker(slope)
        plt.plot(corr_array, mnn_corr_5d[i,i,:].squeeze(), color=color[0])
        plt.plot(corr_array, snn_corr_5d[i,i,:].squeeze(), '--', color=color[1])
        
        #plt.plot(corr_array, mnn_corr_5d[i,i,:].squeeze(), color = '#ff7f0e')
        #plt.plot(corr_array, snn_corr_5d[i,i,:].squeeze(), '--', color='#1f77b4')
        
        if i<10:
            plt.title('$\\bar{{\\sigma}}={}$'.format(std_grid[0,i]), fontsize=8)
        if i % 10 == 0:
            plt.ylabel('$\\bar{{\\mu}}={}$'.format(mean_grid[0,i]), fontsize=8)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(path+'diag_slice_corr_map.png', dpi=300)
    plt.savefig(path+'diag_slice_corr_map.pdf', format='pdf')
    plt.close('all')

if is_plot[1]:
    print('2. Fix input stats to one neuron...')
    i = int(2*10+4)
    print('mean={}, std={}'.format(mean_grid[0,i], std_grid[0,i]))
    plt.figure(figsize=(8,7.5))
    for j in range(mean_grid.shape[1]):
        plt.subplot(11,10, int(j+1))        
        L = 1/20
        slope = (np.sqrt(L)*std_grid[0,j])/(1.0-mean_grid[0,j])
        color = color_picker(slope)
        plt.plot(corr_array, mnn_corr_5d[i,j,:].squeeze(), color=color[0])
        plt.plot(corr_array, snn_corr_5d[i,j,:].squeeze(), '--', color=color[1])
        #plt.plot(corr_array, mnn_corr_5d[i,j,:].squeeze(), color = '#ff7f0e')
        #plt.plot(corr_array, snn_corr_5d[i,j,:].squeeze(), '--', color='#1f77b4')
        if j<10:
            plt.title('$\\bar{{\\sigma}}={}$'.format(std_grid[0,j]), fontsize=8)
        if j % 10 == 0:
            plt.ylabel('$\\bar{{\\mu}}={}$'.format(mean_grid[0,j]), fontsize=8)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(path+'horiz_slice_corr_map.png', dpi=300)
    plt.savefig(path+'horiz_slice_corr_map.pdf', format='pdf')
    plt.close('all')

if is_plot[2]:
    #import matplotlib
    #matplotlib.use('Agg') #support alpha
    print('3. Lower dimensional visualization...')

    #mean_grid = dat['mean_grid']
    #std_grid = dat['std_grid']
    #remove outliers
    corr_err_L2[mean_grid.squeeze()>1,:] = 0
    corr_err_L2[:,mean_grid.squeeze()>1] = 0

    plt.figure()
    plt.hist(corr_err_L2.flatten() , bins=100, range=(0.01, 0.2))    
    plt.savefig('tmp_hist_corr_err.png')
    
    plt.figure()        
    mask = corr_err_L2>0.1
    plt.imshow(mask)
    plt.savefig('tmp_corr_err_mask.png')
    

    plt.figure()
    indx_i, indx_j = np.where(mask)
    
    sizes = mean_grid.flatten()[indx_j]
    sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min())
    sizes *=50
    sizes = sizes.tolist()
    plt.scatter( mean_grid.flatten()[indx_i], std_grid.flatten()[indx_i],s=sizes)
    plt.xlabel('Input current mean')
    plt.ylabel('Input current std')
    plt.savefig('tmp_scatter_of_large_err.png')

    plt.figure()
    plt.hist2d(mean_grid.flatten()[indx_i], std_grid.flatten()[indx_i], bins=4)
    plt.xlabel('Input current mean')
    plt.ylabel('Input current std')
    plt.savefig('tmp_hist2d_large_err.png')

    plt.close('all')

    plt.figure() # parallel axis plot
    #indx_i, indx_j = parameter indices for neuron 1 and 2
    x = mean_grid.squeeze()
    y = std_grid.squeeze()
    plt.scatter(x[indx_i], y[indx_i])
    plt.scatter(x[indx_j], -y[indx_j]) #mirror neuron 2 axis
    print('Total number of points: ', len(indx_i))
    max_err = corr_err_L2[mask].flatten().max()
    min_err = corr_err_L2[mask].flatten().min()
    print('max_err: ', max_err)
    print('min_err: ', min_err)
    
    for k in range(len(indx_i)):
        i = indx_i[k]
        j = indx_j[k]
        print('corr_err_L2: ', corr_err_L2[i,j])
        lw = (corr_err_L2[i,j]-min_err)/(max_err-min_err)*2
        alpha = (corr_err_L2[i,j]-min_err)/(max_err-min_err)
        print('alpha is:', alpha)
        alpha = alpha**2 # make transparent more transparent
        plt.plot([x[i], x[j]], [y[i], -y[j]], alpha=alpha, color='#1f77b4',linewidth=lw)
    plt.ylim([-5,5])
    plt.xlim([-2,3])
    plt.savefig('tmp_para_coord.png',dpi=300)

    plt.close('all')

if is_plot[3]:
    print('4. Effective drive plots')

    corr_err_L2[mean_grid.squeeze()>1,:] = 0
    corr_err_L2[:,mean_grid.squeeze()>1] = 0

    L = 1/20
    Vth = 20

    # let's term it "effective drive"
    drive = (np.sqrt(L)*std_grid)/(L*Vth-mean_grid)
    drive = drive.squeeze()
    
    print('length of drive:' , len(drive))
    print('num of unique drive values:', len(np.unique(drive)))
    print('unique drive values:', np.unique(drive))
    # but drive is not unique!!! multiple (mean,std) can have the same drive
    # I need to calculate the average, and also visualize the std
    # call it 'effective input drive'

    indx_i, indx_j = np.where(corr_err_L2>0.05)
    drive1 = drive + 0.01*np.random.randn(len(drive)) #add random jittering
    drive2 = drive + 0.01*np.random.randn(len(drive)) #add random jittering

    print(mean_grid.squeeze()[indx_i])   
    print(std_grid.squeeze()[indx_i])   
    
    plt.figure()
    print('max corr: ', corr_err_L2.flatten().max())
    plt.scatter(drive1[indx_i], drive2[indx_j], c = corr_err_L2[indx_i,indx_j].flatten())#, s=3.0)
    plt.xlabel('Effective drive 1')
    plt.ylabel('Effective drive 2')
    plt.savefig('tmp_corr_err_vs_drive.png')

    plt.figure()
    plt.scatter(drive[indx_i], corr_err_L2[indx_i,:].max(axis=1))
    #plt.scatter(drive[indx_i], np.mode(corr_err_L2[indx_i,:],axis=1))
    #plt.scatter(drive[indx_i], corr_err_L2[indx_i,:].min(axis=1))
    plt.xlabel('Effective drive')
    plt.ylabel('Max L2 distance')
    plt.savefig('tmp_corr_err_vs_sum_drive.png') 
    
    plt.close('all')

if is_plot[4]:
    ''' Plot convex hull '''
    corr_err_L2[mean_grid.squeeze()>1,:] = 0
    corr_err_L2[:,mean_grid.squeeze()>1] = 0

    L = 1/20
    Vth = 20

    drive = (np.sqrt(L)*std_grid)/(L*Vth-mean_grid)
    drive = drive.squeeze()

    plt.figure()
    L2_thr = np.linspace(0,0.2,6)#[1:]
    L2_thr[0] = -0.1
    for th in L2_thr:
        indx_i, indx_j = np.where(corr_err_L2 > th)
        points = np.zeros( (len(indx_i) ,2))
        points[:,0] = mean_grid.squeeze()[indx_i]
        points[:,1] = std_grid.squeeze()[indx_i]
        #points = np.vstack((points, [1,0])) # manually add a point
        hull = ConvexHull(points)
        hull_vertices = points[hull.vertices]
        
        cmap = matplotlib.cm.viridis
        norm = matplotlib.colors.Normalize(vmin=0, vmax=0.25)
        color = cmap(norm(th))
        
        plt.fill(hull_vertices[:,0], hull_vertices[:,1], color=color, edgecolor='none')

        #plt.plot(points[:,0], points[:,1], '.')
        # focus on marginal for 1 neuron, ignore the other neuron
        # Plot the convex hull
        #for simplex in hull.simplices:
        #    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    plt.savefig('tmp_convex_hull.png')
    plt.close('all')
    # looks kinda bad...

if is_plot[5]:
    ''' good old marginal '''
    # apply mask
    #mean_range = 1.0
    #corr_err_L2[mean_grid.squeeze()>mean_range,:] = 0
    #corr_err_L2[:,mean_grid.squeeze()>mean_range] = 0
    # remove outliers when mean >1 and std<=1.0
    corr_err_L2[50::10,:] = corr_err_L2[52::10,:]
    corr_err_L2[:,50::10] = corr_err_L2[:,52::10]
    corr_err_L2[51::10,:] = corr_err_L2[52::10,:]
    corr_err_L2[:,51::10] = corr_err_L2[:,52::10]

    L = 1/20
    Vth = 20

    drive = (np.sqrt(L)*std_grid)/(L*Vth-mean_grid)
    drive = drive.squeeze()

    max_err = corr_err_L2.max(axis=1).reshape(11,10).T #make y-axis std

    plt.figure(figsize=(4,3.5))
    x = np.linspace(-1,4,11)
    y = np.linspace(0,5,11)[1:]
    #extent=(-1-0.25,1+0.25, 1-0.25,4+0.25)    
    #temp = max_err[np.logical_and(y>0.5,y<4.1), :]
    #temp = max_err[y<4.1, :]
    #extent=(-1-0.25,mean_range+0.25, 0.5-0.25,4+0.25)
    #temp = temp[:,x<mean_range+0.1]
    #plt.imshow(temp, origin='lower', extent=extent)
    extent=(x[0]-0.25, x[-1]+0.25, y[0]-0.25, y[-1]+0.25)
    plt.imshow(max_err, origin='lower', extent=extent, vmin=0,vmax=0.25)
    ycept = lambda gam: gam*2.25/np.sqrt(L)
    plt.plot([1, -1.25], [0, ycept(0.25)], 'k--', linewidth=2)
    plt.plot([1, -1.25], [0, ycept(1.25)], 'k--', linewidth=2)
    plt.xlabel('Input current mean')
    plt.ylabel('Input current std')
    #plt.xlim(-1,1)
    plt.ylim(0.5-0.25,5.25)
    plt.colorbar()
    plt.tight_layout()
    
    plt.savefig(path+'max_err_2d_heat_map.pdf', format='pdf', dpi=300)
    plt.savefig(path+'max_err_2d_heat_map.png')    
    plt.close('all')

if is_plot[6]:
    print('4. Effective drive plots')

    corr_err_L2[mean_grid.squeeze()>1,:] = 0
    corr_err_L2[:,mean_grid.squeeze()>1] = 0
    indx_i, indx_j = np.where(corr_err_L2>0.05)

    L = 1/20
    Vth = 20

    # let's term it "effective drive"
    drive = (np.sqrt(L)*std_grid)/(L*Vth-mean_grid)
    drive = drive.squeeze()
    
    print('length of drive:' , len(drive))
    print('num of unique drive values:', len(np.unique(drive)))
    #print('unique drive values:', np.unique(drive))
    
    plt.figure()
    uni_drive = np.unique(drive)
    uni_drive = uni_drive[~np.isinf(uni_drive)]
    uni_drive = uni_drive[uni_drive>0] # NB this is only defined for mu<1
    print(uni_drive)

    flags = np.zeros(uni_drive.shape)    
    flag = 0
    for i in range(len(uni_drive)-1):
        if uni_drive[i+1]-uni_drive[i] > 0.03: #threshold for combining x-coordinates
            flag += 1 # advance flag
        flags[i+1] = flag
    positions = []
    E = []
    print('flags:', flags)
    for i in range(int(flags[-1])):
        tmp = uni_drive[flags==i] # clustered together data points   
        print(tmp, tmp[0], tmp[-1])
        positions.append(tmp.mean().round(2)) #take average of coordinates
        kk = np.logical_and(drive > (tmp[0]-1e-2) , drive < (tmp[-1]+1e-2))
        print('# entries satisfy condition:', np.sum(kk))
        e = corr_err_L2[kk,:].flatten()
        e = e[np.abs(e)>0.1] # take threshold
        #print(len(e))
        E.append(e)
    print('positions:', positions)
    plt.boxplot(E, positions=positions, widths=0.05, showfliers=False)
    plt.xlabel('Effective drive')
    plt.ylabel('L2 distance')
    plt.xlim([0,2])
    plt.savefig('tmp_eff_drive_boxplot.png')
    plt.close('all')
    # LOOKS terrible
    


    
    