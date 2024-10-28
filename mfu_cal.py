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
    # 后向传播的flops是前向传播的两倍,所以总的flops是前向传播的三倍
    flops_total = flops_forward * 3
    # 计算理论上训练时间
    # flops_total的单位是MACs,所以要转换成TFLOPS(乘以2然后除以1e12)
    flops_total = 2*flops_total / (1e12)
    time_theory = flops_total / FP32_4090
    # 把模型放到GPU上
    model = model.to(device)
    input = input.to(device)
    # 计算实际训练时间
    time_begin = time.time()
    # 模拟一次前向传播和后向传播和参数更新
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    model(input).sum().backward()
    optimizer.zero_grad()
    optimizer.step()
    # 计算实际训练时间
    time_end = time.time()
    time_real = time_end - time_begin
    MFU = time_theory / time_real
    # 把模型放回CPU
    model = model.to("cpu")
    input = input.to("cpu")
    # 清空显存
    torch.cuda.empty_cache()
    print("MFU: " + str(MFU))
    return MFU