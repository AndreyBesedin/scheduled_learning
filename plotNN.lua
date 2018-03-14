-- plot NN
require 'gnuplot'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

netC = nn.Sequential()
netC:add(nn.Linear(2, 5)):add(nn.ReLU(true)):add(nn.Linear(5,4)):add(nn.LogSoftMax())
p, gp = netC:parameters();
gp:zero()

function plot_net_activation(params)
  to_plot = {}
  L = #params/2
  idx_plot = 1
  for l = 1, L do
    for idx_prev = 1, params[2*l-1]:size(2) do
      for idx_post = 1, params[2*l-1]:size(1) do
        x = torch.FloatTensor({l-1, l})
        y = torch.FloatTensor({idx_prev, idx_post})
        val = params[2*l-1][idx_post][idx_prev]
        if val<0 then col = 'rgb "blue"' else col = 'rgb "red"' end
        to_plot[idx_plot] = {x, y, 'with lines lc ' .. col .. ' lw ' .. math.abs(val)}
        idx_plot = idx_plot + 1
      end
    end
  end
  return to_plot
end
