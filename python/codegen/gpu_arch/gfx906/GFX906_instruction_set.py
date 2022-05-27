#generated by instruction_parser.py
from typing import Any, Union
from python.codegen.gpu_arch.gpu_instruct import inst_base, inst_caller_base, instruction_type
from python.codegen.gpu_arch.gpu_data_types import *

class vop2_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST:Union[regVar,None,Any], SRC0:Union[regVar,None,Any], SRC1:Union[regVar,None,Any], MODIFIERS:str): 
		super().__init__(instruction_type.VOP2, INSTRUCTION)
		self.DST = DST 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
		self.MODIFIERS = MODIFIERS 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST,self.SRC0,self.SRC1]) 
		return f"{self.label} {', '.join(map(str, args_l))} {self.MODIFIERS}" 
class vop2_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def v_fmac_f32(self, vdst:regVar, src0:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,lds_direct_t,const,int], vsrc1:regVar):
		return self.ic_pb(vop2_base('v_fmac_f32', vdst, src0, vsrc1, ''))
	def v_fmac_f32_dpp(self, vdst:regVar, vsrc0:regVar, vsrc1:regVar, MODIFIERS:str=''):
		""":param str MODIFIERS: dpp_ctrl row_mask bank_mask bound_ctrl"""
		return self.ic_pb(vop2_base('v_fmac_f32_dpp', vdst, vsrc0, vsrc1, MODIFIERS))
	def v_xnor_b32(self, vdst:regVar, src0:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,lds_direct_t,const,int], vsrc1:regVar):
		return self.ic_pb(vop2_base('v_xnor_b32', vdst, src0, vsrc1, ''))
	def v_xnor_b32_dpp(self, vdst:regVar, vsrc0:regVar, vsrc1:regVar, MODIFIERS:str=''):
		""":param str MODIFIERS: dpp_ctrl row_mask bank_mask bound_ctrl"""
		return self.ic_pb(vop2_base('v_xnor_b32_dpp', vdst, vsrc0, vsrc1, MODIFIERS))
	def v_xnor_b32_sdwa(self, vdst:regVar, src0:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], src1:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], MODIFIERS:str=''):
		""":param str MODIFIERS: dst_sel dst_unused src0_sel src1_sel"""
		return self.ic_pb(vop2_base('v_xnor_b32_sdwa', vdst, src0, src1, MODIFIERS))
class vop3_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST:Union[regVar,None,Any], SRC0:Union[regVar,None,Any], SRC1:Union[regVar,None,Any], MODIFIERS:str): 
		super().__init__(instruction_type.VOP3, INSTRUCTION)
		self.DST = DST 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
		self.MODIFIERS = MODIFIERS 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST,self.SRC0,self.SRC1]) 
		return f"{self.label} {', '.join(map(str, args_l))} {self.MODIFIERS}" 
class vop3_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def v_fmac_f32_e64(self, vdst:regVar, src0:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,lds_direct_t,const], src1:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], MODIFIERS:str=''):
		""":param str MODIFIERS: clamp omod"""
		return self.ic_pb(vop3_base('v_fmac_f32_e64', vdst, src0, src1, MODIFIERS))
	def v_xnor_b32_e64(self, vdst:regVar, src0:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,lds_direct_t,const], src1:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const]):
		return self.ic_pb(vop3_base('v_xnor_b32_e64', vdst, src0, src1, ''))
class vop3p_base(inst_base): 
	def __init__(self, INSTRUCTION:str, DST:Union[regVar,None,Any], SRC0:Union[regVar,None,Any], SRC1:Union[regVar,None,Any], SRC2:Union[regVar,None,Any], MODIFIERS:str): 
		super().__init__(instruction_type.VOP3P, INSTRUCTION)
		self.DST = DST 
		self.SRC0 = SRC0 
		self.SRC1 = SRC1 
		self.SRC2 = SRC2 
		self.MODIFIERS = MODIFIERS 
	def __str__(self): 
		args_l = filter(None.__ne__, [self.DST,self.SRC0,self.SRC1,self.SRC2]) 
		return f"{self.label} {', '.join(map(str, args_l))} {self.MODIFIERS}" 
class vop3p_instr_caller(inst_caller_base): 
	def __init__(self, insturction_container) -> None:
     		super().__init__(insturction_container)
	def v_dot2_f32_f16(self, vdst:regVar, src0:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,lds_direct_t,const], src1:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], src2:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], MODIFIERS:str=''):
		""":param str MODIFIERS: neg_lo neg_hi clamp"""
		return self.ic_pb(vop3p_base('v_dot2_f32_f16', vdst, src0, src1, src2, MODIFIERS))
	def v_dot2_i32_i16(self, vdst:regVar, src0:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,lds_direct_t,iconst,ival_t], src1:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,iconst,ival_t], src2:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], MODIFIERS:str=''):
		""":param str MODIFIERS: clamp"""
		return self.ic_pb(vop3p_base('v_dot2_i32_i16', vdst, src0, src1, src2, MODIFIERS))
	def v_dot2_u32_u16(self, vdst:regVar, src0:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,lds_direct_t,iconst,ival_t], src1:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,iconst,ival_t], src2:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], MODIFIERS:str=''):
		""":param str MODIFIERS: clamp"""
		return self.ic_pb(vop3p_base('v_dot2_u32_u16', vdst, src0, src1, src2, MODIFIERS))
	def v_dot4_i32_i8(self, vdst:regVar, src0:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,lds_direct_t,const], src1:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], src2:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], MODIFIERS:str=''):
		""":param str MODIFIERS: clamp"""
		return self.ic_pb(vop3p_base('v_dot4_i32_i8', vdst, src0, src1, src2, MODIFIERS))
	def v_dot4_u32_u8(self, vdst:regVar, src0:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,lds_direct_t,const], src1:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], src2:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], MODIFIERS:str=''):
		""":param str MODIFIERS: clamp"""
		return self.ic_pb(vop3p_base('v_dot4_u32_u8', vdst, src0, src1, src2, MODIFIERS))
	def v_dot8_i32_i4(self, vdst:regVar, src0:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,lds_direct_t,const], src1:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], src2:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], MODIFIERS:str=''):
		""":param str MODIFIERS: clamp"""
		return self.ic_pb(vop3p_base('v_dot8_i32_i4', vdst, src0, src1, src2, MODIFIERS))
	def v_dot8_u32_u4(self, vdst:regVar, src0:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,lds_direct_t,const], src1:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], src2:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], MODIFIERS:str=''):
		""":param str MODIFIERS: clamp"""
		return self.ic_pb(vop3p_base('v_dot8_u32_u4', vdst, src0, src1, src2, MODIFIERS))
	def v_fma_mix_f32(self, vdst:regVar, src0:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,lds_direct_t,const], src1:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], src2:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], MODIFIERS:str=''):
		""":param str MODIFIERS: m_op_sel m_op_sel_hi clamp"""
		return self.ic_pb(vop3p_base('v_fma_mix_f32', vdst, src0, src1, src2, MODIFIERS))
	def v_fma_mixhi_f16(self, vdst:regVar, src0:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,lds_direct_t,const], src1:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], src2:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], MODIFIERS:str=''):
		""":param str MODIFIERS: m_op_sel m_op_sel_hi clamp"""
		return self.ic_pb(vop3p_base('v_fma_mixhi_f16', vdst, src0, src1, src2, MODIFIERS))
	def v_fma_mixlo_f16(self, vdst:regVar, src0:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,lds_direct_t,const], src1:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], src2:Union[regVar,flat_scratch,xnack_mask,VCC_reg,TTMP_reg,M0_reg,EXEC_reg,SCC_bit,const], MODIFIERS:str=''):
		""":param str MODIFIERS: m_op_sel m_op_sel_hi clamp"""
		return self.ic_pb(vop3p_base('v_fma_mixlo_f16', vdst, src0, src1, src2, MODIFIERS))
