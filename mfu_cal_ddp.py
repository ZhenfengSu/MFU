'''
计算单张卡的前向推理MFU和吞吐量
'''
import torch
from get_model import flops_deit
import time
FP32_4090 = 82.6 # 单位是TFLOPS
# 给定model和batch_size,计算出mfu
def mfu(model, batch_size,model_info_map):
    print("evaluating MFU")
    print("batch_size: " + str(batch_size))
    device = torch.device("cuda")
    input = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)
    hidden_dim = model_info_map["hidden_dim"]
    patch_size = model_info_map["patch_size"]
    num_classes = 1000
    depth = model_info_map["depth"]
    # 计算flops(flops_deit是bs=1的flops)
    flops_forward = flops_deit(hidden_dim, patch_size, num_classes, depth)* batch_size
    # 计算理论上训练时间
    # flops_total的单位是MACs,所以要转换成TFLOPS(乘以2然后除以1e12)
    flops_total = 2*flops_forward / (1e12)
    time_theory = flops_total / FP32_4090
    # 把模型放到GPU上
    model = model.to(device)
    input = input.to(device)
    # 计算实际训练时间
    model.train()
    time_list = []
    # 模拟一次前向传播
    for i in range(40):
        print("iter: " + str(i))
        time_begin = time.time()
        output = model(input)
        torch.cuda.synchronize()   
        time_end = time.time() 
        time_list.append(time_end - time_begin)
    # 计算实际训练时间
    # 去掉前10次的时间
    time_list = time_list[10:]
    time_real = sum(time_list) / len(time_list)
    MFU = time_theory / time_real
    # 把模型放回CPU
    model = model.to("cpu")
    input = input.to("cpu")
    # 清空显存
    torch.cuda.empty_cache()
    print("MFU: " + str(MFU))
    return MFU

# 给定model和batch_size,计算出mfu和throughput
def mfu_throughput(model,batch_size,model_info_map):
    print("evaluating MFU and throughput")
    print("batch_size: " + str(batch_size))
    device = torch.device("cuda")
    input = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)
    hidden_dim = model_info_map["hidden_dim"]
    patch_size = model_info_map["patch_size"]
    num_classes = 1000
    depth = model_info_map["depth"]
    # 计算flops(flops_deit是bs=1的flops)
    flops_forward = flops_deit(hidden_dim, patch_size, num_classes, depth)* batch_size
    # flops_total的单位是MACs,所以要转换成TFLOPS(乘以2然后除以1e12)
    flops_total = 2*flops_forward / (1e12)
    # 计算throughput
    # 把模型放到GPU上
    model = model.to(device)
    input = input.to(device)
    # 计算实际训练时间
    model.train()
    time_list = []
    # 模拟一次前向传播
    for i in range(40):
        print("iter: " + str(i))
        time_begin = time.time()
        output = model(input)
        torch.cuda.synchronize()   
        time_end = time.time() 
        time_list.append(time_end - time_begin)
    # 计算实际训练时间
    # 去掉前10次的时间
    time_list = time_list[10:]
    
    # images nums
    images_num = batch_size * len(time_list)
    time_real_sum = sum(time_list) 
    time_real_avg = time_real_sum / len(time_list)
    throughput = images_num / time_real_sum
    
    MFU = flops_total * throughput / (FP32_4090*batch_size)
    # 把模型放回CPU
    model = model.to("cpu")
    input = input.to("cpu")
    # 清空显存
    torch.cuda.empty_cache()
    print("MFU: " + str(MFU))
    
    return MFU, throughput