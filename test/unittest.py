from igemm import *
import math

def get_default_mc():
    return mc_asm_printer_t(mc_emit_to_string_t(), amdgpu_arch_config_t(None))

def unittest_share_memory():
    v_dst = sym_t('v_dst')
    v_sld = sym_t('v_sld')
    v_src = sym_t('v_src')
    v_sst = sym_t('v_sst')

    mc = get_default_mc()
    sldx2 = inst_ds_read2_likely_t(mc, 4, 16, 1030)
    mc.emit(sldx2(v_dst(), v_sld()))
    #print(mc.emitter.get_buffer())

    sstx2 = inst_ds_write2_likely_t(mc, 4, 8, 512)
    mc.emit(sstx2(v_sst(), v_src(), 256))
    print(mc.emitter.get_buffer())

def unittest_coalescing_store():

    mc = get_default_mc()
    ctm = ctrl_thread_mapping_t()
    ctm.thread_lengths = [2,2,1,1,4,4]
    ctm.cluster_lengths = [1,1,4,4,4,4]

    ctrl = ctrl_coalescing_store_t()
    ctrl.ctm = ctm
    ctrl.coalescing_groups = 4
    ctrl.data_byte = 4

    ctrl.vector_write_out = 1
    ctrl.block_size = 256

    coalescing_store = igemm_coalescing_store_t(mc, ctrl)


    mc.emit(coalescing_store.init_co_lds_offset('v_co_sst', 'v_co_sld', 'v_gemm_im', 'v_gemm_in', 'v0', 'v_tmp'))
    mc.emit(coalescing_store.init_co_sub_m_index('v_co_sub_m_index', 'v_tid', 'v_tmp'))
    mc.emit(coalescing_store('v_c', 'v_co_sst', 'v_co_sld', 's_p_out', 'v_out_offset', 's_out_offset', 's_gemm_m_stride', 's_tmp'))
    print(mc.emitter.get_buffer())



def unittest_coalescing_store_m1_m0():
    mc = get_default_mc()
    ctm = ctrl_thread_mapping_t()
    ctm.thread_lengths = [2,2,1,1,4,4]
    ctm.cluster_lengths = [1,1,4,4,4,4]

    ctrl = ctrl_coalescing_store_t()
    ctrl.ctm = ctm
    ctrl.coalescing_groups = 4
    ctrl.data_byte = 4

    ctrl.vector_write_out = 1
    ctrl.block_size = 256
    ctrl.gemm_m_order = IGEMM_COALESCING_GEMM_M_ORDER_M1_M0
    ctrl.gemm_m_m0_m1 = [4, 32]

    ctrl.adjust_optimal_coalescing_groups()

    m_index_per_group       = ctrl.get_m_index_per_group()
    m_index_per_group_m1_m0 = ctrl.get_m_index_per_group_m1_m0()
    for ig in range(len(m_index_per_group)):
        for ic in range(len(m_index_per_group[ig])):
            print(f"ig:{ig} ic:{ic}, m0_m1: {m_index_per_group[ig][ic]}")
            print("    |" + " ".join( f"{ctrl.get_m0_m1_index(x)}" for x in m_index_per_group[ig][ic]))
    print("")
    for ig in range(len(m_index_per_group)):
        for ic in range(len(m_index_per_group[ig])):
            print(f"ig:{ig} ic:{ic}, m1_m0: {m_index_per_group_m1_m0[ig][ic]}")
            print("    |" + " ".join( f"{ctrl.get_m0_m1_index(x)}" for x in m_index_per_group_m1_m0[ig][ic]))


    coalescing_store = igemm_coalescing_store_t(mc, ctrl)

    mc.emit(coalescing_store.init_co_lds_offset('v_co_sst', 'v_co_sld', 'v_gemm_im', 'v_gemm_in', 'v0', 'v_tmp'))
    mc.emit(coalescing_store.init_co_sub_m_index('v_co_sub_m_index', 'v_tid', 'v_tmp'))
    mc.emit(coalescing_store('v_c', 'v_co_sst', 'v_co_sld', 's_p_out', 'v_out_offset', 's_out_offset', 's_gemm_m0_stride', 's_gemm_m1_stride', 's_tmp'))
    print(mc.emitter.get_buffer())

def unittest_coalescing_store_m1_m0_xdlops():
    macro_tile_m = 128
    macro_tile_n = 256
    block_size = 256

    mc = get_default_mc()
    cxm = get_ctrl_xdlops_mapping_fp32(macro_tile_m, macro_tile_n, block_size // 64)

    ctrl = ctrl_coalescing_store_xdlops_t()
    ctrl.cxm = cxm
    ctrl.coalescing_groups = 2
    ctrl.data_byte = 4

    ctrl.vector_write_out = 1
    ctrl.block_size = block_size
    ctrl.gemm_m_order = IGEMM_COALESCING_GEMM_M_ORDER_M1_M0
    ctrl.gemm_m_m0_m1 = [4, macro_tile_m // 4] # similar to non-xdlops

    ctrl.adjust_optimal_coalescing_groups()

    m_index_per_group       = ctrl.get_m_index_per_group()
    m_index_per_group_m1_m0 = ctrl.get_m_index_per_group_m1_m0()
    for ig in range(len(m_index_per_group)):
        for ic in range(len(m_index_per_group[ig])):
            print(f"ig:{ig} ic:{ic}, m0_m1: {m_index_per_group[ig][ic]}")
            print("    |" + " ".join( f"{ctrl.get_m0_m1_index(x)}" for x in m_index_per_group[ig][ic]))
    print("")
    for ig in range(len(m_index_per_group)):
        for ic in range(len(m_index_per_group[ig])):
            print(f"ig:{ig} ic:{ic}, m1_m0: {m_index_per_group_m1_m0[ig][ic]}")
            print("    |" + " ".join( f"{ctrl.get_m0_m1_index(x)}" for x in m_index_per_group_m1_m0[ig][ic]))


    #coalescing_store = igemm_coalescing_store_t(mc, ctrl)

    #mc.emit(coalescing_store.init_co_lds_offset('v_co_sst', 'v_co_sld', 'v_gemm_im', 'v_gemm_in', 'v0', 'v_tmp'))
    #mc.emit(coalescing_store.init_co_sub_m_index('v_co_sub_m_index', 'v_tid', 'v_tmp'))
    #mc.emit(coalescing_store('v_c', 'v_co_sst', 'v_co_sld', 's_p_out', 'v_out_offset', 's_out_offset', 's_gemm_m0_stride', 's_gemm_m1_stride', 's_tmp'))
    #print(mc.emitter.get_buffer())

def unittest_xdlops_mapping():
    for xdlops_mapping in ctrl_xdlops_mapping_fp32:
        print(xdlops_mapping.serialize())


def unittest_coalescing_store_m1_m0_xdlops_iterate():
    for xdlops_mapping in ctrl_xdlops_mapping_fp32:
        # max_possible_groups = xdlops_mapping.wave_repeat_m * xdlops_mapping.wave_step_m * xdlops_mapping.lanegroup_m_per_wave() * xdlops_mapping.lanegroup_m_per_block() * xdlops_mapping.lanegroup_m_per_thread()
        max_possible_groups = xdlops_mapping.wave_repeat_m * xdlops_mapping.wave_step_m * xdlops_mapping.lanegroup_m_per_wave() * \
                xdlops_mapping.lanegroup_m_per_block() * xdlops_mapping.lanegroup_m_per_thread() \
                    // 4
        cgroup_list = [2**x for x in range(0, int(math.log2(max_possible_groups)) + 1)]
        print(f"[<<<<<<]max_possible_groups:{max_possible_groups}, cgroup_list:{cgroup_list}, {xdlops_mapping.serialize()}")
        for cgroups in cgroup_list:
            mc = get_default_mc()
            print(f"[------] groups:{cgroups}")
            ctrl = ctrl_coalescing_store_xdlops_t()
            ctrl.cxm = xdlops_mapping
            ctrl.coalescing_groups = cgroups
            ctrl.data_byte = 4

            ctrl.vector_write_out = 1
            ctrl.block_size = xdlops_mapping.block_size()
            # ctrl.gemm_m_order = IGEMM_COALESCING_GEMM_M_ORDER_M1_M0
            ctrl.gemm_m_m0_m1 = [4, xdlops_mapping.macro_tile_m // 4] # similar to non-xdlops

            ctrl.adjust_optimal_coalescing_groups()



            coalescing_store = igemm_coalescing_store_xdlops_t(mc, ctrl)

            m_index_per_group       = ctrl.get_m_index_per_group()
            m_index_per_group_m1_m0 = ctrl.get_m_index_per_group_m1_m0()

            # def get_co_sub_m():
            #     ttm = ctrl.get_transposed_thread_mapping()
            #     g_mr, g_ms, g_mw, g_mb, g_mt = ctrl.get_subgroups()
            #     l_mr, l_ms, l_mw, l_mb, l_mt = ctrl.get_subgroup_length()
            #     n_mc = ctrl.cxm.lanegroup_m_per_cluster()       # this iteration is among different thread
            #     n_ml = ctrl.cxm.block_m_per_lanegroup()         # this iteration is among different thread
            #     n_mv = ctrl.cxm.waves_per_m()                   # this iteration is among different thread
            #     #print("<+> g_mr:{}, g_ms:{}, g_mw:{}, g_mb:{}, g_mt:{}  ==  l_mr:{}, l_ms:{}, l_mw:{}, l_mb:{}, l_mt:{} | n_mc:{}, n_ml:{}, n_mv:{}".format(
            #     #        g_mr, g_ms, g_mw, g_mb, g_mt, l_mr, l_ms, l_mw, l_mb, l_mt, n_mc, n_ml, n_mv))
            #     sub_m_index = [0] * ttm.c_m0()
            #     for ic in range(ttm.c_m0()):
            #         nic = ic
            #         x_mc = nic % n_mc; nic = nic // n_mc
            #         x_ml = nic % n_ml; nic = nic // n_ml
            #         x_mb = nic % l_mb; nic = nic // l_mb
            #         x_mw = nic % l_mw; nic = nic // l_mw
            #         x_ms = nic % l_ms; nic = nic // l_ms
            #         x_mv = nic % n_mv; nic = nic // n_mv
            #         x_mr = nic % l_mr
            #         # print("    +-> x_mc:{}, x_ml:{}, x_mb:{}, x_mw:{}, x_ms:{}, x_mv:{}, x_mr:{}".format(x_mc, x_ml, x_mb, x_mw, x_ms, x_mv, x_mr))
            #         sub_m = x_mr * n_mv + x_mv
            #         sub_m = sub_m * (g_ms * l_ms) + x_ms
            #         sub_m = sub_m * (l_mw * g_mw) + x_mw
            #         sub_m = sub_m * (g_mb * l_mb) + x_mb
            #         sub_m = sub_m * n_ml + x_ml
            #         sub_m = sub_m * n_mc + x_mc
            #         sub_m = sub_m * 4
            #         sub_m_index[ic] = sub_m
            #     return sub_m_index

            def get_sliced_sub_m_list():
                # from m_index_per_group[0]
                sliced_sub_m_list = list()
                m_start_index = 0
                for ic in range(len(m_index_per_group[0])):
                    sliced_sub_m_list.append(m_index_per_group[0][ic][m_start_index])
                return sliced_sub_m_list

            print(f"<sub_m_index>:{ctrl.get_co_sub_m_index()}")
            for ig in range(len(m_index_per_group)):
                for ic in range(len(m_index_per_group[ig])):
                    print(f"ig:{ig} ic:{ic}, m0_m1: {m_index_per_group[ig][ic]}")
                    print("    |" + " ".join( f"{ctrl.get_m0_m1_index(x)}" for x in m_index_per_group[ig][ic]))

            mc.emit(coalescing_store.init_co_sub_m_index('v_co_sub_m_index', 'v_tid', 'v_tmp6'))
            mc.emit(';@@@@ ------------------------------------------------')
            mc.emit(coalescing_store('a_c', 'v_c', 'v_co_sst', 'v_co_sld',
                    's_p_out', 'v_out_offset', 's_out_offset' if 0 else None, None, 's_gemm_m1_stride', 's_tmp4', 'v_store_flag'))
            mc.emit(';XXXX ------------------------------------------------')
            print(mc.emitter.get_buffer())
            print("------------------------------------------------------------")


            # assert co_sub_m along returned m_index_per_group[0], aka first group.
            assert ctrl.get_co_sub_m_index() == get_sliced_sub_m_list()

            # print("")
            # for ig in range(len(m_index_per_group)):
            #     for ic in range(len(m_index_per_group[ig])):
            #         print(f"ig:{ig} ic:{ic}, m1_m0: {m_index_per_group_m1_m0[ig][ic]}")
            #         print("    |" + " ".join( f"{ctrl.get_m0_m1_index(x)}" for x in m_index_per_group_m1_m0[ig][ic]))




def unittest_thread_mapping():
    mc = get_default_mc()
    ctm = ctrl_thread_mapping_t()
    ctm.thread_lengths = [2,2,1,1,4,4]
    ctm.cluster_lengths = [1,1,4,4,4,4]
    thread_mapping = igemm_thread_mapping_t(mc, ctm)
    mc.emit(thread_mapping( 'v_gemm_in', 'v_gemm_im', 'v_tid_shifter', 'v_tmp'))
    print(mc.emitter.get_buffer())

def run_all_unittest():
    # unittest_share_memory()
    #unittest_coalescing_store()
    #unittest_coalescing_store_m1_m0()
    #unittest_coalescing_store_m1_m0_xdlops()
    #unittest_xdlops_mapping()
    unittest_coalescing_store_m1_m0_xdlops_iterate()
    # unittest_thread_mapping()

if __name__ == '__main__':
    run_all_unittest()