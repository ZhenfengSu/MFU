from mfu_cal import mfu
# # min_batch_size = 1
# # max_batch_size = 256, 选择256是4090最大只能放得下256

# # 给定model和model_info_map,返回最优的batch_size
# def bs_search(model,model_info_map):
#     # 先计算出bs=1时的MFU
#     MFU_bs1 = mfu(model, 1, model_info_map)
#     # 计算出bs=256时的MFU
#     MFU_bs256 = mfu(model, 256, model_info_map)
#     # 二分查找
    
def binary_search_max_bs(start=1, end=32, model=None,model_info_map=None):  
    """  
    使用二分搜索查找函数在[start, end]区间内的最大值。  
    
    参数:  
    func -- 目标函数，只接受整数作为输入。  
    start -- 搜索区间的起始整数。  
    end -- 搜索区间的结束整数。  
    
    返回值:  
    包含最大值和对应输入的元组 (max_value, corresponding_input)。  
    """  
    while start < end:  
        mid1 = start + (end - start) // 2  
        mid2 = mid1 + 1  
        
        # 计算中间两个点的值  
        f_mid1 = mfu(model=model,batch_size = mid1, model_info_map=model_info_map)  
        f_mid2 = mfu(model=model,batch_size = mid2, model_info_map=model_info_map)  
        
        if f_mid1 < f_mid2:  
            start = mid2  # 保持搜索范围在右侧  
        else:  
            end = mid1  # 保持搜索范围在左侧  

    max_value = mfu(model=model,batch_size = start, model_info_map=model_info_map)
    return max_value, start  