import torch

class MomentConv2d(torch.nn.Module):
    __constant__ = ['in_channels', 'out_channels','kernel_size', 'stride']
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super(MomentConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        #pytorch default 2d convolution, but limit in/out channel size to 1
        #this is a temporary work-around
        self.conv2d = [torch.nn.Conv2d(1, 1, kernel_size, stride=stride, bias=False) for i in range(out_channels)]
        
    def forward(self, u, C):
        #STEP 1: convolve mean
        batch_size, in_channels, height, width = u.shape
        
        #merge in_channels to the batch dimension
        u = u.view(batch_size*in_channels, 1, height, width)
        
        for i in range(self.out_channels):
            tmp = self.conv2d[i](u) #convolve using each kernel one by one
            if i==0: #initialize empty array
                out_mean = torch.empty( batch_size, self.out_channels, tmp.shape[2], tmp.shape[3] )
            
            tmp = tmp.view(batch_size, in_channels, tmp.shape[2], tmp.shape[3]) #recover and sum over in_channels
            tmp = torch.sum(tmp, dim=1)#, keepdim=True)
            out_mean[:,i,:,:] = tmp
        
        #STEP 2: convolve covariance
        #assume that covariance has shape: batch x channel x (HxW) x (HxW) 
        batch_size, in_channels, height, width, _, _ = C.shape
        
        #merge in_channels, height, width to batch dimension
        C = C.view(batch_size*in_channels*height*width, 1, height, width) 
            
        for i in range(self.out_channels):
            
            #convolve once            
            tmp = self.conv2d[i](C) 
            out_h = tmp.shape[2]
            out_w = tmp.shape[3]
            
            if i==0: #initialize empty array
                out_cov = torch.empty( batch_size, self.out_channels, out_h, out_w, out_h, out_w )
            
            #recover the dimensions
            tmp = tmp.view(batch_size, in_channels, height, width ,out_h, out_w)
            
            #transpose the covariance 'matrix'
            tmp = torch.permute(tmp, (0,1,4,5,2,3))  
            
            #merge again to the batch_dimension
            #have to use reshape here instead of view due to the way stride is stored in memory
            tmp = tmp.reshape(batch_size*in_channels*out_h*out_w, 1, height, width) 
            
            #convolve a second time
            tmp = self.conv2d[i](tmp)
            
            #recover dimensions
            tmp = tmp.view(batch_size, in_channels, out_h,out_w, out_h,out_w)
            
            #transpose the covariance 'matrix' back
            tmp = torch.permute(tmp, (0,1,4,5,2,3))  
            
            #sum over in_channels
            tmp = torch.sum(tmp, dim=1)#, keepdim=True)
            out_cov[:,i,:,:,:,:] = tmp
        
        return out_mean, out_cov

if __name__=='__main__':    
    in_channels = 5
    out_channels = 3
    batch_size = 30
    kernel_size = 3
    stride = 2
    
    u = torch.randn(batch_size,in_channels,10,10) #a random image
    C = torch.randn(batch_size,in_channels,10,10,10,10) #a test array for covaraince ignoring positive semi-definiteness
    
    mconv = MomentConv2d(10,8,kernel_size,stride)
    
    out_mean, out_cov = mconv(u, C)