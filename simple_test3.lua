--[[
    Test with supplementary weight for back-propagation, updated on validation set.
   Those weights are updated directly during training of the classifier, based on it's error on validation set.
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
  lrW = 0.01,
  nb_classes = 4,
  load_pretrained_classifier = true,
  load_pretrained_weighter = true,
  train = 'full',
  cuda = true,
  dataset = 'quaters', -- MNIST 
  optimizeW = true,
  reinitW = false,
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

--------------------------------------------------------------------------------------------------------------------------------
-- DEFINING THE WEIGHT GENERATOR
--------------------------------------------------------------------------------------------------------------------------------
if opt.load_pretrained_weighter then
  netW = torch.load('./pretrained_models/W_init_ones.t7')
  pW, gpW = netC:getParameters()
else
  netW = nn.Sequential()
  netW:add(nn.Linear(opt.nb_classes,20)):add(nn.ReLU(true)):add(nn.Linear(20,20)):add(nn.ReLU(true)):add(nn.Linear(20,p:size(1))):add(nn.Sigmoid())
  pW, gpW = netW:getParameters()
end
critC = nn.ClassNLLCriterion(); 
critW = nn.MSECriterion()
--------------------------------------------------------------------------------------------------------------------------------
-- SETTING TO CUDA IF NEEDED
--------------------------------------------------------------------------------------------------------------------------------
if opt.cuda then
  require 'cunn'
  require 'cudnn'
  critC = critC:cuda(); critW = critW:cuda()
  netC = netC:cuda(); netW = netW:cuda()
else
  critC = critC:float(); critW = critW:float()
  netC = netC:float(); netW = netW:float()
end  

stream = true
env_count = 0
W = torch.ones(pC:size(1)); 
res_conf = torch.FloatTensor(opt.nb_env) 
while env_count < opt.nb_env do
  current_env = getEnv(opt)
  env_count = env_count + 1
  print('Current environment: '); print(current_env:reshape(1,4))
  if opt.reinitW then
    W = torch.ones(pC:size(1)); 
  end
  if opt.cuda == true then W =  W:cuda() end
  for idx_batch = 1, opt.batches_per_env do
    batch = generate_2D_quaters({opt.batchSize, 2}, current_env, opt)
    pW, gpW = netW:getParameters()
    pC, gpC = netC:getParameters()
    gpC:zero()
    output = netC:forward(batch.data)               -- (2)
    df_dtheta = critC:backward(output, batch.labels)
    netC:backward(batch.data, df_dtheta)   -- (3)
    if opt.optimizeW == true then
      netC_temp = netC:clone(); pC_temp, gpC_temp = netC_temp:getParameters()
      --gpC:clamp(-10,10)
      pC_temp:add(-opt.lrC, torch.cmul(W, gpC));      -- (4)
      val_indices = torch.randperm(valset.data:size(1))[{{1, opt.valBatchSize}}]
      batch_val = getBatch(valset, val_indices, opt)       -- (5)
      output_val = netC_temp:forward(batch_val.data)  -- (6)
      df_dtheta = critC:backward(output_val, batch_val.labels)
      netC_temp:backward(batch_val.data, df_dtheta); gpC_temp:clamp(-10,10)
      W:add(-opt.lrW*opt.lrC*torch.cmul(gpC_temp, gpC))
      --dW = -opt.lrC*torch.cmul(gpC_temp, gpC)                 -- (7.2)
    end
    W:clamp(0,1)
    pC:add(-opt.lrC, torch.cmul(W, gpC))
  --  print("Sum of parameters of W: " .. W:sum())
  end
  print("Sum of parameters of W: " .. W:sum())
  confusion = test_classifier(netC, testset, opt); print(confusion)
  res_conf[env_count] = confusion.totalValid
end
    
    
    
