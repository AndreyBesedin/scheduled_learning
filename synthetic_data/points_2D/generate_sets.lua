require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
opt = {}
opt.manualSeed = torch.random(1, 10000) -- fix seed
torch.manualSeed(opt.manualSeed)

train_size = 100000
class_size = 50000
val_size = 20000
test_size = 20000
box_size = 100
function generate_2D_quaters(dsize, box_size)
  local data = {}
  data.data = torch.zeros(dsize, 2); data.data = data.data:uniform(-box_size, box_size); data.labels = torch.zeros(dsize)
  for idx = 1, dsize do
    local x1 = data.data[idx][1]; local x2 = data.data[idx][2]
    data.labels[idx] = x1>=0 and x2>0 and 1 or x1>0 and x2<=0 and 2 or x1<=0 and x2<0 and 3 or x1<=0 and x2>=0 and 4
  end
  return data
end

function generate_2D_quaters_partial(class_size, box_size, env)
  --env = {e1,e2,e3,e4}, ei = 1 if class is present and 0 otherwise
  local data = {}
  local eps = 1e-15
  labels = {}; idx_lab = 1; for idx = 1, #env do if env[idx]==1 then labels[idx_lab] = idx; idx_lab = idx_lab+1 end; end
  data.data = torch.zeros(class_size*#labels, 2)
  data.labels = torch.zeros(class_size*#labels)
  for idx_class = 1, #labels do
    if labels[idx_class] == 1 then 
      box = {0, box_size, eps, box_size}
    elseif labels[idx_class] == 2 then 
      box = {eps, box_size, -box_size, 0}
    elseif labels[idx_class] == 3 then
      box = {-box_size, 0, -box_size, -eps}
    else
      box = {-box_size, 0, 0, box_size}
    end
    data.data[{{1 + (idx_class-1)*class_size, idx_class*class_size},{1}}] = data.data[{{1 + (idx_class-1)*class_size, idx_class*class_size},{1}}]:uniform(box[1], box[2]); 
    data.data[{{1 + (idx_class-1)*class_size, idx_class*class_size},{2}}] = data.data[{{1 + (idx_class-1)*class_size, idx_class*class_size},{2}}]:uniform(box[3], box[4]); 
    data.labels[{{1 + (idx_class-1)*class_size, idx_class*class_size}}]:fill(labels[idx_class])
  end
  return data
end

--train = generate_2D_quaters(train_size, box_size)
--val = generate_2D_quaters(val_size, box_size)
--test = generate_2D_quaters(test_size, box_size)
--os.execute('mkdir quaters_2D')
--torch.save('quaters_2D/train.t7', train)
--torch.save('quaters_2D/val.t7', val)
--torch.save('quaters_2D/test.t7', test)
for 
train1 = generate_2D_quaters_partial(class_size, box_size, {1,0,0,0}); torch.save('quaters_2D/train1.t7', train1)
train2 = generate_2D_quaters_partial(class_size, box_size, {0,1,0,0}); torch.save('quaters_2D/train2.t7', train2)
train3 = generate_2D_quaters_partial(class_size, box_size, {0,0,1,0}); torch.save('quaters_2D/train3.t7', train3)
train4 = generate_2D_quaters_partial(class_size, box_size, {0,0,0,1}); torch.save('quaters_2D/train4.t7', train4)

train12 = generate_2D_quaters_partial(class_size, box_size, {1,1,0,0}); torch.save('quaters_2D/train12.t7', train12)
train23 = generate_2D_quaters_partial(class_size, box_size, {0,1,1,0); torch.save('quaters_2D/train23.t7', train23)
train13 = generate_2D_quaters_partial(class_size, box_size, {1,0,1,0}); torch.save('quaters_2D/train13.t7', train13)
train14 = generate_2D_quaters_partial(class_size, box_size, {1,0,0,1}); torch.save('quaters_2D/train14.t7', train14)
train24 = generate_2D_quaters_partial(class_size, box_size, {0,1,0,1); torch.save('quaters_2D/train24.t7', train24)
train34 = generate_2D_quaters_partial(class_size, box_size, {0,0,1,1); torch.save('quaters_2D/train34.t7', train34)

train123 = generate_2D_quaters_partial(class_size, box_size, {1,1,1,0}); torch.save('quaters_2D/train123.t7', train123)
train124 = generate_2D_quaters_partial(class_size, box_size, {1,1,0,1}); torch.save('quaters_2D/train124.t7', train124)
train134 = generate_2D_quaters_partial(class_size, box_size, {1,0,1,1); torch.save('quaters_2D/train134.t7', train134)
train234 = generate_2D_quaters_partial(class_size, box_size, {0,1,1,1); torch.save('quaters_2D/train234.t7', train234)

train1234 = generate_2D_quaters_partial(class_size, box_size, {1,1,1,1}); torch.save('quaters_2D/train1234.t7', train1234)




