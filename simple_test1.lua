require 'torch'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'

torch.setdefaulttensortype('torch.FloatTensor')
opt = {
  batchSize = 1000,
  epochs = 1000,
  lr = 0.01,
  nb_classes = 4,
  load_pretrained_classifier = true,
  load_pretrained_weighter = true,
  train = 'full'
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
  p, gp = netC:getParameters()
else
  netC = nn.Sequential()
  netC:add(nn.Linear(2,10)):add(nn.ReLU(true)):add(nn.Linear(10,10)):add(nn.ReLU(true)):add(nn.Linear(10,4)):add(nn.LogSoftMax())
  netC = netC:cuda()
  p, gp = netC:getParameters()
  p = p:normal()
end
crit = nn.ClassNLLCriterion();

--------------------------------------------------------------------------------------------------------------------------------
-- DEFINING THE WEIGHT GENERATOR
--------------------------------------------------------------------------------------------------------------------------------
if opt.load_pretrained_weighter then
  netW = torch.load('./pretrained_models/W_init_ones.t7')
  pW, gpW = netC:getParameters()
else
  netW = nn.Sequential()
  netW:add(nn.Linear(opt.nb_classes,20)):add(nn.ReLU(true)):add(nn.Linear(20,20)):add(nn.ReLU(true)):add(nn.Linear(20,p:size(1))):add(nn.Sigmoid())
  netW = netW:cuda()
  pW, gpW = netW:getParameters()
end
bs = opt.batchSize

function generateEnv(nb_classes, bs)
  local data = torch.rand(bs, nb_classes)
  return data:ge(0.5):float():cuda()
end
  
function getBatch(data, indices_)
  local batch = {}
  batch.data = data.data:index(1, indices_:long())
  batch.labels = data.labels:index(1, indices_:long())
  return batch
end

function test_classifier(C, data, opt)
  local confusion = optim.ConfusionMatrix(opt.nb_classes)
  confusion:zero()
  for idx = 1, data.data:size(1), opt.batchSize do
    --xlua.progress(idx, opt.testSize)
    indices_ = torch.range(idx, math.min(idx + opt.batchSize, data.data:size(1)))
    local batch = getBatch(data, indices_:long())
    local y = C:forward(batch.data:cuda())
    y = y:float()
    _, y_max = y:max(2)
    confusion:batchAdd(y_max:squeeze():float(), batch.labels:float())
  end
  confusion:updateValids()
  return confusion  
end

function train_classifier(C, data, opt)
  local fx = function(x)
    if x ~= p then p:copy(x) end
    gp:zero()
    local output = C:forward(input)
    y = output:float()
    _, y_max = y:max(2)
    local errM = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    C:backward(input, df_do)
    confusion_train:batchAdd(y_max:squeeze():float(), label:float())
    return errM, gp
  end
  input = torch.CudaTensor(opt.batchSize, 2)
  label = torch.CudaTensor(opt.batchSize)
  criterion = crit:cuda()
  local indices_rand = torch.randperm(data.data:size(1))
  confusion_train = optim.ConfusionMatrix(opt.nb_classes)
  confusion_train:zero()
  idx_test = 0
  confusion_test_ = {}
  for i = 1, math.floor(data.data:size(1)/opt.batchSize) do 
    --xlua.progress(i, math.floor(data.data:size(1)/opt.batchSize))
    indices_ = indices_rand[{{1+(i-1)*opt.batchSize, i*opt.batchSize}}]
    batch = getBatch(data, indices_:long())
    input:copy(batch.data)
    label:copy(batch.labels)
--    optim.sgd(fx, p, config)
    f_, dfdx = fx(p)
    p:add(-opt.lr, dfdx)
    --C_model:clearState()
    p, gp = C:getParameters()
  end
  --print('Training set confusion matrix: ')
  --print(confusion_train)
  return C, confusion_train
end

critW = nn.MSECriterion()
function train_W(W, opt)
  local fx = function(x)
    if x ~= pW then pW:copy(x) end
    gpW:zero()
    out_star = W:forward(input)
    local errM = criterion:forward(out_star, out_)
    local df_do = criterion:backward(out_star, out_)
    W:backward(input, df_do)
    return errM, gpW
  end
  input = torch.CudaTensor(opt.batchSize, opt.nb_classes)
  out_ = torch.ones(opt.batchSize, p:size(1)):cuda()
  criterion = critW:cuda()
  for i = 1, 1e+5 do 
    --xlua.progress(i, math.floor(data.data:size(1)/opt.batchSize))
    batch = generateEnv(opt.nb_classes, opt.batchSize)
    input:copy(batch)
--    optim.sgd(fx, p, configW)
    f_, dfdx = fx(pW)
    pW:add(-opt.lr, dfdx)
    pW, gpW = W:getParameters()
    print('Parameters sum: ' .. out_star:sum()/1000)
  end
  return W
end
configW = {}
configW.learningRate = opt.lr
netW = train_W(netW, opt)
-- config = {}
-- config.learningRate = opt.lr
-- conf = test_classifier(netC, testset,opt); print(conf)
-- 
-- for epoch = 1, opt.epochs do
--   print('Epoch number: ' .. epoch .. ', Learning rate: ' .. opt.lr)
-- --  if epoch%100==0 then opt.lr = opt.lr/1.5 end
--   train_classifier(netC, trainset, opt)
--   conf = test_classifier(netC, testset,opt); print(conf)
-- end   
--     
    
    
    
    
