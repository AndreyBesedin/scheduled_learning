--[[
    Test with supplementary regularization step on back-propagation.
    Weights of the classifier are mixed-updated on every step first on stream and then on validation set 
   ]]
require 'torch'
require 'optim'
require 'nn'
dofile('./lib_stream.lua')
torch.setdefaulttensortype('torch.FloatTensor')
opt = {
  batchSize = 60,
  valBatchSize = 60,
  batches_per_env = 1000,
  epochs = 1000,
  lrC = 0.0005,
  nb_classes = 4,
  load_pretrained_classifier = true,
  load_pretrained_weighter = true,
  train = 'full',
  cuda = true,
  dataset = 'quaters', -- MNIST 
  reg_on_validation = true,
  nb_env = 200,
}
opt.manualSeed = torch.random(1, 10000) -- fix seed
torch.manualSeed(opt.manualSeed)

data_path = './synthetic_data/points_2D/quaters_2D/'
trainset = torch.load(data_path .. 'train123.t7')
valset = torch.load(data_path .. 'val.t7')
testset = torch.load(data_path .. 'test.t7')
--------------------------------------------------------------------------------------------------------------------------------
-- DEFINING THE CLASSIFIER
--------------------------------------------------------------------------------------------------------------------------------
if opt.load_pretrained_classifier then
  netC = torch.load('./pretrained_models/C_full_quaters2D.t7')
  pC, gpC = netC:getParameters()
else
  netC = nn.Sequential()
  netC:add(nn.Linear(2,10)):add(nn.ReLU(true)):add(nn.Linear(10,10)):add(nn.ReLU(true)):add(nn.Linear(10,4)):add(nn.LogSoftMax())
  pC, gpC = netC:getParameters()
  pC = pC:normal()
end

critC = nn.ClassNLLCriterion(); 
--------------------------------------------------------------------------------------------------------------------------------
-- SETTING TO CUDA IF NEEDED
--------------------------------------------------------------------------------------------------------------------------------
if opt.cuda then
  require 'cunn'
  require 'cudnn'
  critC = critC:cuda();
  netC = netC:cuda(); 
else
  critC = critC:float(); 
  netC = netC:float(); 
end  

stream = true
env_count = 0
res_conf = torch.FloatTensor(opt.nb_env) 
while env_count < opt.nb_env do
  current_env = getEnv(opt)
  env_count = env_count + 1
  print('Current environment: '); print(current_env:reshape(1,4))
  --print('Sum of parameters of W(e): ' .. W:sum())
  for idx_batch = 1, opt.batches_per_env do
    batch = generate_2D_quaters({opt.batchSize, 2}, current_env, opt)
    pC, gpC = netC:getParameters()
    gpC:zero()
    output = netC:forward(batch.data)               -- (2)
    df_dtheta = critC:backward(output, batch.labels)
    netC:backward(batch.data, df_dtheta)   -- (3)
    pC:add(-opt.lrC, gpC)
    if opt.reg_on_validation == true then
      gpC:zero()
      val_indices = torch.randperm(valset.data:size(1))[{{1, opt.valBatchSize}}]
      batch_val = getBatch(valset, val_indices, opt)       -- (5)
      output_val = netC:forward(batch_val.data)  -- (6)
      df_dtheta = critC:backward(output_val, batch_val.labels)
      netC:backward(batch_val.data, df_dtheta); -- gpC:clamp(-5,5)
      pC:add(-opt.lrC, gpC)
    end
  end
  confusion = test_classifier(netC, testset, opt); print(confusion)
  res_conf[env_count] = confusion.totalValid
end
    
    
    
