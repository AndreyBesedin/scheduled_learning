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
  lrW = 10,
  nb_classes = 10,
  load_pretrained_classifier = false,
  load_pretrained_weighter = true,
  train = 'full',
  cuda = false,
  dataset = 'MNIST', -- MNIST 
  optimizeW = true,
  nb_env = 200
}
opt.manualSeed = torch.random(1, 10000) -- fix seed
torch.manualSeed(opt.manualSeed)

data_path = './MNIST/t7_files/'
trainset = torch.load(data_path .. 'trainset.t7')
valset = torch.load(data_path .. 'validation.t7')
testset = torch.load(data_path .. 'testset.t7')
--------------------------------------------------------------------------------------------------------------------------------
-- DEFINING THE CLASSIFIER
--------------------------------------------------------------------------------------------------------------------------------
netC = nn.Sequential()
netC:add(nn.SpatialConvolution(1,8, 4, 4, 2, 2, 1, 1)):add(nn.ReLU(true))
netC:add(nn.SpatialMaxPooling(2, 2, 2,2))  
netC:add(nn.SpatialConvolution(8, 16, 3, 3, 2, 2, 1, 1)):add(nn.ReLU(true))
netC:add(nn.SpatialMaxPooling(2, 2, 2,2)):add(nn.View(64))
netC:add(nn.Linear(64,10)):add(nn.ReLU(true))
netC:add(nn.Linear(10,10)):add(nn.LogSoftMax())
pC, gpC = netC:getParameters()
pC = pC:normal()

optimState = {
  learningRate = 0.001,
}
confusion_train = optim.ConfusionMatrix(10)

critC = nn.ClassNLLCriterion(); 
for epoch = 1, opt.epochs do
  local indices = torch.randperm(trainset.data:size(1))
  for idx = 1, trainset.data:size(1), opt.batchSize do
    xlua.progress(idx, trainset.data:size(1))
    local feval = function(x)
      if x ~= pC then pC:copy(x) end
      gpC:zero()
      local outputs = netC:forward(batch.data)
      local f = critC:forward(outputs, batch.labels)
      local df_do = critC:backward(outputs, batch.labels)
      netC:backward(batch.data, df_do)
      confusion:batchAdd(outputs, batch.labels)
      return f,gpC
    end    
    batch = getBatch(trainset, indices[{{idx, math.min(idx + opt.batchSize - 1, trainset.data:size(1))}}]:long())
    optim.adam(feval, pC, optimState)
    if (idx-1)%(100*opt.batchSize)==0 then confusion = test_classifier(netC, testset, opt); print(confusion) end
  end
  confusion_train:updateValids()
  print('Training confusion: '); print(confusion_train)
  print('Epoch ' .. epoch .. ' is over, testing')
  confusion = test_classifier(netC, testset, opt); print(confusion)
end