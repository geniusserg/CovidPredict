úÊ
Ñ¡
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8®Î

Adam/dense_88/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_88/bias/v
y
(Adam/dense_88/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_88/bias/v*
_output_shapes
:*
dtype0

Adam/dense_88/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*'
shared_nameAdam/dense_88/kernel/v

*Adam/dense_88/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_88/kernel/v*
_output_shapes

:/*
dtype0

Adam/dense_87/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_87/bias/v
y
(Adam/dense_87/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_87/bias/v*
_output_shapes
:/*
dtype0

Adam/dense_87/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*'
shared_nameAdam/dense_87/kernel/v

*Adam/dense_87/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_87/kernel/v*
_output_shapes

://*
dtype0

Adam/dense_88/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_88/bias/m
y
(Adam/dense_88/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_88/bias/m*
_output_shapes
:*
dtype0

Adam/dense_88/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*'
shared_nameAdam/dense_88/kernel/m

*Adam/dense_88/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_88/kernel/m*
_output_shapes

:/*
dtype0

Adam/dense_87/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_87/bias/m
y
(Adam/dense_87/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_87/bias/m*
_output_shapes
:/*
dtype0

Adam/dense_87/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*'
shared_nameAdam/dense_87/kernel/m

*Adam/dense_87/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_87/kernel/m*
_output_shapes

://*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
r
dense_88/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_88/bias
k
!dense_88/bias/Read/ReadVariableOpReadVariableOpdense_88/bias*
_output_shapes
:*
dtype0
z
dense_88/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/* 
shared_namedense_88/kernel
s
#dense_88/kernel/Read/ReadVariableOpReadVariableOpdense_88/kernel*
_output_shapes

:/*
dtype0
r
dense_87/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namedense_87/bias
k
!dense_87/bias/Read/ReadVariableOpReadVariableOpdense_87/bias*
_output_shapes
:/*
dtype0
z
dense_87/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
://* 
shared_namedense_87/kernel
s
#dense_87/kernel/Read/ReadVariableOpReadVariableOpdense_87/kernel*
_output_shapes

://*
dtype0

serving_default_input_32Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ/
ü
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_32dense_87/kerneldense_87/biasdense_88/kerneldense_88/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1295823

NoOpNoOp
ò%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*­%
value£%B % B%
§
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias*
 
0
1
!2
"3*
 
0
1
!2
"3*
* 
°
#non_trainable_variables

$layers
%metrics
&layer_regularization_losses
'layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
(trace_0
)trace_1
*trace_2
+trace_3* 
6
,trace_0
-trace_1
.trace_2
/trace_3* 
* 

0iter

1beta_1

2beta_2
	3decay
4learning_ratemPmQ!mR"mSvTvU!vV"vW*

5serving_default* 
* 
* 
* 

6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

;trace_0* 

<trace_0* 

0
1*

0
1*
* 

=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Btrace_0* 

Ctrace_0* 
_Y
VARIABLE_VALUEdense_87/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_87/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

!0
"1*

!0
"1*
* 

Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*

Itrace_0* 

Jtrace_0* 
_Y
VARIABLE_VALUEdense_88/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_88/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

K0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
L	variables
M	keras_api
	Ntotal
	Ocount*

N0
O1*

L	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_87/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_87/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_88/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_88/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_87/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_87/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_88/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_88/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ö
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_87/kernel/Read/ReadVariableOp!dense_87/bias/Read/ReadVariableOp#dense_88/kernel/Read/ReadVariableOp!dense_88/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_87/kernel/m/Read/ReadVariableOp(Adam/dense_87/bias/m/Read/ReadVariableOp*Adam/dense_88/kernel/m/Read/ReadVariableOp(Adam/dense_88/bias/m/Read/ReadVariableOp*Adam/dense_87/kernel/v/Read/ReadVariableOp(Adam/dense_87/bias/v/Read/ReadVariableOp*Adam/dense_88/kernel/v/Read/ReadVariableOp(Adam/dense_88/bias/v/Read/ReadVariableOpConst* 
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_1296017
õ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_87/kerneldense_87/biasdense_88/kerneldense_88/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_87/kernel/mAdam/dense_87/bias/mAdam/dense_88/kernel/mAdam/dense_88/bias/mAdam/dense_87/kernel/vAdam/dense_87/bias/vAdam/dense_88/kernel/vAdam/dense_88/bias/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_1296084êó
È	
ö
E__inference_dense_88_layer_call_and_return_conditional_losses_1295674

inputs0
matmul_readvariableop_resource:/-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
N
¡
#__inference__traced_restore_1296084
file_prefix2
 assignvariableop_dense_87_kernel://.
 assignvariableop_1_dense_87_bias:/4
"assignvariableop_2_dense_88_kernel:/.
 assignvariableop_3_dense_88_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: "
assignvariableop_9_total: #
assignvariableop_10_count: <
*assignvariableop_11_adam_dense_87_kernel_m://6
(assignvariableop_12_adam_dense_87_bias_m:/<
*assignvariableop_13_adam_dense_88_kernel_m:/6
(assignvariableop_14_adam_dense_88_bias_m:<
*assignvariableop_15_adam_dense_87_kernel_v://6
(assignvariableop_16_adam_dense_87_bias_v:/<
*assignvariableop_17_adam_dense_88_kernel_v:/6
(assignvariableop_18_adam_dense_88_bias_v:
identity_20¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ò

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ø	
valueî	Bë	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_dense_87_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_87_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_88_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_88_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp*assignvariableop_11_adam_dense_87_kernel_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp(assignvariableop_12_adam_dense_87_bias_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_dense_88_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp(assignvariableop_14_adam_dense_88_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_87_kernel_vIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_87_bias_vIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_88_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_88_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ñ
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_20IdentityIdentity_19:output:0^NoOp_1*
T0*
_output_shapes
: Þ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_20Identity_20:output:0*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¸
Ù
6__inference_Flatten2Layers_Mixed_layer_call_fn_1295836

inputs
unknown://
	unknown_0:/
	unknown_1:/
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295681o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ/: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
«
H
,__inference_flatten_23_layer_call_fn_1295892

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_23_layer_call_and_return_conditional_losses_1295645`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs

¾
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295681

inputs"
dense_87_1295659://
dense_87_1295661:/"
dense_88_1295675:/
dense_88_1295677:
identity¢ dense_87/StatefulPartitionedCall¢ dense_88/StatefulPartitionedCall½
flatten_23/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_23_layer_call_and_return_conditional_losses_1295645
 dense_87/StatefulPartitionedCallStatefulPartitionedCall#flatten_23/PartitionedCall:output:0dense_87_1295659dense_87_1295661*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_87_layer_call_and_return_conditional_losses_1295658
 dense_88/StatefulPartitionedCallStatefulPartitionedCall)dense_87/StatefulPartitionedCall:output:0dense_88_1295675dense_88_1295677*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_88_layer_call_and_return_conditional_losses_1295674x
IdentityIdentity)dense_88/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_87/StatefulPartitionedCall!^dense_88/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ/: : : : 2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2D
 dense_88/StatefulPartitionedCall dense_88/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
¾
×
"__inference__wrapped_model_1295632
input_32N
<flatten2layers_mixed_dense_87_matmul_readvariableop_resource://K
=flatten2layers_mixed_dense_87_biasadd_readvariableop_resource:/N
<flatten2layers_mixed_dense_88_matmul_readvariableop_resource:/K
=flatten2layers_mixed_dense_88_biasadd_readvariableop_resource:
identity¢4Flatten2Layers_Mixed/dense_87/BiasAdd/ReadVariableOp¢3Flatten2Layers_Mixed/dense_87/MatMul/ReadVariableOp¢4Flatten2Layers_Mixed/dense_88/BiasAdd/ReadVariableOp¢3Flatten2Layers_Mixed/dense_88/MatMul/ReadVariableOpv
%Flatten2Layers_Mixed/flatten_23/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ/   
'Flatten2Layers_Mixed/flatten_23/ReshapeReshapeinput_32.Flatten2Layers_Mixed/flatten_23/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/°
3Flatten2Layers_Mixed/dense_87/MatMul/ReadVariableOpReadVariableOp<flatten2layers_mixed_dense_87_matmul_readvariableop_resource*
_output_shapes

://*
dtype0Ï
$Flatten2Layers_Mixed/dense_87/MatMulMatMul0Flatten2Layers_Mixed/flatten_23/Reshape:output:0;Flatten2Layers_Mixed/dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/®
4Flatten2Layers_Mixed/dense_87/BiasAdd/ReadVariableOpReadVariableOp=flatten2layers_mixed_dense_87_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0Ð
%Flatten2Layers_Mixed/dense_87/BiasAddBiasAdd.Flatten2Layers_Mixed/dense_87/MatMul:product:0<Flatten2Layers_Mixed/dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
"Flatten2Layers_Mixed/dense_87/ReluRelu.Flatten2Layers_Mixed/dense_87/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/°
3Flatten2Layers_Mixed/dense_88/MatMul/ReadVariableOpReadVariableOp<flatten2layers_mixed_dense_88_matmul_readvariableop_resource*
_output_shapes

:/*
dtype0Ï
$Flatten2Layers_Mixed/dense_88/MatMulMatMul0Flatten2Layers_Mixed/dense_87/Relu:activations:0;Flatten2Layers_Mixed/dense_88/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4Flatten2Layers_Mixed/dense_88/BiasAdd/ReadVariableOpReadVariableOp=flatten2layers_mixed_dense_88_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%Flatten2Layers_Mixed/dense_88/BiasAddBiasAdd.Flatten2Layers_Mixed/dense_88/MatMul:product:0<Flatten2Layers_Mixed/dense_88/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
IdentityIdentity.Flatten2Layers_Mixed/dense_88/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp5^Flatten2Layers_Mixed/dense_87/BiasAdd/ReadVariableOp4^Flatten2Layers_Mixed/dense_87/MatMul/ReadVariableOp5^Flatten2Layers_Mixed/dense_88/BiasAdd/ReadVariableOp4^Flatten2Layers_Mixed/dense_88/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ/: : : : 2l
4Flatten2Layers_Mixed/dense_87/BiasAdd/ReadVariableOp4Flatten2Layers_Mixed/dense_87/BiasAdd/ReadVariableOp2j
3Flatten2Layers_Mixed/dense_87/MatMul/ReadVariableOp3Flatten2Layers_Mixed/dense_87/MatMul/ReadVariableOp2l
4Flatten2Layers_Mixed/dense_88/BiasAdd/ReadVariableOp4Flatten2Layers_Mixed/dense_88/BiasAdd/ReadVariableOp2j
3Flatten2Layers_Mixed/dense_88/MatMul/ReadVariableOp3Flatten2Layers_Mixed/dense_88/MatMul/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
"
_user_specified_name
input_32

Ü
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295887

inputs9
'dense_87_matmul_readvariableop_resource://6
(dense_87_biasadd_readvariableop_resource:/9
'dense_88_matmul_readvariableop_resource:/6
(dense_88_biasadd_readvariableop_resource:
identity¢dense_87/BiasAdd/ReadVariableOp¢dense_87/MatMul/ReadVariableOp¢dense_88/BiasAdd/ReadVariableOp¢dense_88/MatMul/ReadVariableOpa
flatten_23/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ/   r
flatten_23/ReshapeReshapeinputsflatten_23/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
dense_87/MatMul/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

://*
dtype0
dense_87/MatMulMatMulflatten_23/Reshape:output:0&dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
dense_87/BiasAdd/ReadVariableOpReadVariableOp(dense_87_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0
dense_87/BiasAddBiasAdddense_87/MatMul:product:0'dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/b
dense_87/ReluReludense_87/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
dense_88/MatMul/ReadVariableOpReadVariableOp'dense_88_matmul_readvariableop_resource*
_output_shapes

:/*
dtype0
dense_88/MatMulMatMuldense_87/Relu:activations:0&dense_88/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_88/BiasAdd/ReadVariableOpReadVariableOp(dense_88_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_88/BiasAddBiasAdddense_88/MatMul:product:0'dense_88/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_88/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
NoOpNoOp ^dense_87/BiasAdd/ReadVariableOp^dense_87/MatMul/ReadVariableOp ^dense_88/BiasAdd/ReadVariableOp^dense_88/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ/: : : : 2B
dense_87/BiasAdd/ReadVariableOpdense_87/BiasAdd/ReadVariableOp2@
dense_87/MatMul/ReadVariableOpdense_87/MatMul/ReadVariableOp2B
dense_88/BiasAdd/ReadVariableOpdense_88/BiasAdd/ReadVariableOp2@
dense_88/MatMul/ReadVariableOpdense_88/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
Ä

*__inference_dense_87_layer_call_fn_1295907

inputs
unknown://
	unknown_0:/
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_87_layer_call_and_return_conditional_losses_1295658o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
¸
Ù
6__inference_Flatten2Layers_Mixed_layer_call_fn_1295849

inputs
unknown://
	unknown_0:/
	unknown_1:/
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295748o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ/: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
¿
c
G__inference_flatten_23_layer_call_and_return_conditional_losses_1295645

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ/   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs

Ü
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295868

inputs9
'dense_87_matmul_readvariableop_resource://6
(dense_87_biasadd_readvariableop_resource:/9
'dense_88_matmul_readvariableop_resource:/6
(dense_88_biasadd_readvariableop_resource:
identity¢dense_87/BiasAdd/ReadVariableOp¢dense_87/MatMul/ReadVariableOp¢dense_88/BiasAdd/ReadVariableOp¢dense_88/MatMul/ReadVariableOpa
flatten_23/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ/   r
flatten_23/ReshapeReshapeinputsflatten_23/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
dense_87/MatMul/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

://*
dtype0
dense_87/MatMulMatMulflatten_23/Reshape:output:0&dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
dense_87/BiasAdd/ReadVariableOpReadVariableOp(dense_87_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0
dense_87/BiasAddBiasAdddense_87/MatMul:product:0'dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/b
dense_87/ReluReludense_87/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
dense_88/MatMul/ReadVariableOpReadVariableOp'dense_88_matmul_readvariableop_resource*
_output_shapes

:/*
dtype0
dense_88/MatMulMatMuldense_87/Relu:activations:0&dense_88/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_88/BiasAdd/ReadVariableOpReadVariableOp(dense_88_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_88/BiasAddBiasAdddense_88/MatMul:product:0'dense_88/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_88/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
NoOpNoOp ^dense_87/BiasAdd/ReadVariableOp^dense_87/MatMul/ReadVariableOp ^dense_88/BiasAdd/ReadVariableOp^dense_88/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ/: : : : 2B
dense_87/BiasAdd/ReadVariableOpdense_87/BiasAdd/ReadVariableOp2@
dense_87/MatMul/ReadVariableOpdense_87/MatMul/ReadVariableOp2B
dense_88/BiasAdd/ReadVariableOpdense_88/BiasAdd/ReadVariableOp2@
dense_88/MatMul/ReadVariableOpdense_88/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
þ
Ê
%__inference_signature_wrapper_1295823
input_32
unknown://
	unknown_0:/
	unknown_1:/
	unknown_2:
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallinput_32unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_1295632o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ/: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
"
_user_specified_name
input_32
¾
Û
6__inference_Flatten2Layers_Mixed_layer_call_fn_1295772
input_32
unknown://
	unknown_0:/
	unknown_1:/
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_32unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295748o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ/: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
"
_user_specified_name
input_32
¿
c
G__inference_flatten_23_layer_call_and_return_conditional_losses_1295898

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ/   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs


ö
E__inference_dense_87_layer_call_and_return_conditional_losses_1295918

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
¾
Û
6__inference_Flatten2Layers_Mixed_layer_call_fn_1295692
input_32
unknown://
	unknown_0:/
	unknown_1:/
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_32unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295681o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ/: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
"
_user_specified_name
input_32
Ä

*__inference_dense_88_layer_call_fn_1295927

inputs
unknown:/
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_88_layer_call_and_return_conditional_losses_1295674o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs

¾
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295748

inputs"
dense_87_1295737://
dense_87_1295739:/"
dense_88_1295742:/
dense_88_1295744:
identity¢ dense_87/StatefulPartitionedCall¢ dense_88/StatefulPartitionedCall½
flatten_23/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_23_layer_call_and_return_conditional_losses_1295645
 dense_87/StatefulPartitionedCallStatefulPartitionedCall#flatten_23/PartitionedCall:output:0dense_87_1295737dense_87_1295739*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_87_layer_call_and_return_conditional_losses_1295658
 dense_88/StatefulPartitionedCallStatefulPartitionedCall)dense_87/StatefulPartitionedCall:output:0dense_88_1295742dense_88_1295744*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_88_layer_call_and_return_conditional_losses_1295674x
IdentityIdentity)dense_88/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_87/StatefulPartitionedCall!^dense_88/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ/: : : : 2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2D
 dense_88/StatefulPartitionedCall dense_88/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs


ö
E__inference_dense_87_layer_call_and_return_conditional_losses_1295658

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs

À
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295802
input_32"
dense_87_1295791://
dense_87_1295793:/"
dense_88_1295796:/
dense_88_1295798:
identity¢ dense_87/StatefulPartitionedCall¢ dense_88/StatefulPartitionedCall¿
flatten_23/PartitionedCallPartitionedCallinput_32*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_23_layer_call_and_return_conditional_losses_1295645
 dense_87/StatefulPartitionedCallStatefulPartitionedCall#flatten_23/PartitionedCall:output:0dense_87_1295791dense_87_1295793*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_87_layer_call_and_return_conditional_losses_1295658
 dense_88/StatefulPartitionedCallStatefulPartitionedCall)dense_87/StatefulPartitionedCall:output:0dense_88_1295796dense_88_1295798*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_88_layer_call_and_return_conditional_losses_1295674x
IdentityIdentity)dense_88/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_87/StatefulPartitionedCall!^dense_88/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ/: : : : 2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2D
 dense_88/StatefulPartitionedCall dense_88/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
"
_user_specified_name
input_32
È	
ö
E__inference_dense_88_layer_call_and_return_conditional_losses_1295937

inputs0
matmul_readvariableop_resource:/-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs

À
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295787
input_32"
dense_87_1295776://
dense_87_1295778:/"
dense_88_1295781:/
dense_88_1295783:
identity¢ dense_87/StatefulPartitionedCall¢ dense_88/StatefulPartitionedCall¿
flatten_23/PartitionedCallPartitionedCallinput_32*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_23_layer_call_and_return_conditional_losses_1295645
 dense_87/StatefulPartitionedCallStatefulPartitionedCall#flatten_23/PartitionedCall:output:0dense_87_1295776dense_87_1295778*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_87_layer_call_and_return_conditional_losses_1295658
 dense_88/StatefulPartitionedCallStatefulPartitionedCall)dense_87/StatefulPartitionedCall:output:0dense_88_1295781dense_88_1295783*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_88_layer_call_and_return_conditional_losses_1295674x
IdentityIdentity)dense_88/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_87/StatefulPartitionedCall!^dense_88/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ/: : : : 2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2D
 dense_88/StatefulPartitionedCall dense_88/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
"
_user_specified_name
input_32
ç.

 __inference__traced_save_1296017
file_prefix.
*savev2_dense_87_kernel_read_readvariableop,
(savev2_dense_87_bias_read_readvariableop.
*savev2_dense_88_kernel_read_readvariableop,
(savev2_dense_88_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_87_kernel_m_read_readvariableop3
/savev2_adam_dense_87_bias_m_read_readvariableop5
1savev2_adam_dense_88_kernel_m_read_readvariableop3
/savev2_adam_dense_88_bias_m_read_readvariableop5
1savev2_adam_dense_87_kernel_v_read_readvariableop3
/savev2_adam_dense_87_bias_v_read_readvariableop5
1savev2_adam_dense_88_kernel_v_read_readvariableop3
/savev2_adam_dense_88_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ï

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ø	
valueî	Bë	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_87_kernel_read_readvariableop(savev2_dense_87_bias_read_readvariableop*savev2_dense_88_kernel_read_readvariableop(savev2_dense_88_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_87_kernel_m_read_readvariableop/savev2_adam_dense_87_bias_m_read_readvariableop1savev2_adam_dense_88_kernel_m_read_readvariableop/savev2_adam_dense_88_bias_m_read_readvariableop1savev2_adam_dense_87_kernel_v_read_readvariableop/savev2_adam_dense_87_bias_v_read_readvariableop1savev2_adam_dense_88_kernel_v_read_readvariableop/savev2_adam_dense_88_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *"
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapest
r: ://:/:/:: : : : : : : ://:/:/:://:/:/:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

://: 

_output_shapes
:/:$ 

_output_shapes

:/: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

://: 

_output_shapes
:/:$ 

_output_shapes

:/: 

_output_shapes
::$ 

_output_shapes

://: 

_output_shapes
:/:$ 

_output_shapes

:/: 

_output_shapes
::

_output_shapes
: "µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*±
serving_default
A
input_325
serving_default_input_32:0ÿÿÿÿÿÿÿÿÿ/<
dense_880
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:k
Á
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias"
_tf_keras_layer
<
0
1
!2
"3"
trackable_list_wrapper
<
0
1
!2
"3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
#non_trainable_variables

$layers
%metrics
&layer_regularization_losses
'layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object

(trace_0
)trace_1
*trace_2
+trace_32¢
6__inference_Flatten2Layers_Mixed_layer_call_fn_1295692
6__inference_Flatten2Layers_Mixed_layer_call_fn_1295836
6__inference_Flatten2Layers_Mixed_layer_call_fn_1295849
6__inference_Flatten2Layers_Mixed_layer_call_fn_1295772¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z(trace_0z)trace_1z*trace_2z+trace_3
ù
,trace_0
-trace_1
.trace_2
/trace_32
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295868
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295887
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295787
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295802¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z,trace_0z-trace_1z.trace_2z/trace_3
ÎBË
"__inference__wrapped_model_1295632input_32"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 

0iter

1beta_1

2beta_2
	3decay
4learning_ratemPmQ!mR"mSvTvU!vV"vW"
	optimizer
,
5serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ð
;trace_02Ó
,__inference_flatten_23_layer_call_fn_1295892¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z;trace_0

<trace_02î
G__inference_flatten_23_layer_call_and_return_conditional_losses_1295898¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z<trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
î
Btrace_02Ñ
*__inference_dense_87_layer_call_fn_1295907¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zBtrace_0

Ctrace_02ì
E__inference_dense_87_layer_call_and_return_conditional_losses_1295918¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zCtrace_0
!://2dense_87/kernel
:/2dense_87/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
î
Itrace_02Ñ
*__inference_dense_88_layer_call_fn_1295927¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zItrace_0

Jtrace_02ì
E__inference_dense_88_layer_call_and_return_conditional_losses_1295937¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zJtrace_0
!:/2dense_88/kernel
:2dense_88/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
K0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
6__inference_Flatten2Layers_Mixed_layer_call_fn_1295692input_32"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
6__inference_Flatten2Layers_Mixed_layer_call_fn_1295836inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
6__inference_Flatten2Layers_Mixed_layer_call_fn_1295849inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
6__inference_Flatten2Layers_Mixed_layer_call_fn_1295772input_32"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¢B
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295868inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¢B
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295887inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¤B¡
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295787input_32"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¤B¡
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295802input_32"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÍBÊ
%__inference_signature_wrapper_1295823input_32"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_flatten_23_layer_call_fn_1295892inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_flatten_23_layer_call_and_return_conditional_losses_1295898inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_dense_87_layer_call_fn_1295907inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_dense_87_layer_call_and_return_conditional_losses_1295918inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_dense_88_layer_call_fn_1295927inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_dense_88_layer_call_and_return_conditional_losses_1295937inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
N
L	variables
M	keras_api
	Ntotal
	Ocount"
_tf_keras_metric
.
N0
O1"
trackable_list_wrapper
-
L	variables"
_generic_user_object
:  (2total
:  (2count
&:$//2Adam/dense_87/kernel/m
 :/2Adam/dense_87/bias/m
&:$/2Adam/dense_88/kernel/m
 :2Adam/dense_88/bias/m
&:$//2Adam/dense_87/kernel/v
 :/2Adam/dense_87/bias/v
&:$/2Adam/dense_88/kernel/v
 :2Adam/dense_88/bias/vÁ
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295787l!"=¢:
3¢0
&#
input_32ÿÿÿÿÿÿÿÿÿ/
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295802l!"=¢:
3¢0
&#
input_32ÿÿÿÿÿÿÿÿÿ/
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¿
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295868j!";¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ/
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¿
Q__inference_Flatten2Layers_Mixed_layer_call_and_return_conditional_losses_1295887j!";¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ/
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
6__inference_Flatten2Layers_Mixed_layer_call_fn_1295692_!"=¢:
3¢0
&#
input_32ÿÿÿÿÿÿÿÿÿ/
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
6__inference_Flatten2Layers_Mixed_layer_call_fn_1295772_!"=¢:
3¢0
&#
input_32ÿÿÿÿÿÿÿÿÿ/
p

 
ª "ÿÿÿÿÿÿÿÿÿ
6__inference_Flatten2Layers_Mixed_layer_call_fn_1295836]!";¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ/
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
6__inference_Flatten2Layers_Mixed_layer_call_fn_1295849]!";¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ/
p

 
ª "ÿÿÿÿÿÿÿÿÿ
"__inference__wrapped_model_1295632r!"5¢2
+¢(
&#
input_32ÿÿÿÿÿÿÿÿÿ/
ª "3ª0
.
dense_88"
dense_88ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_87_layer_call_and_return_conditional_losses_1295918\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ/
 }
*__inference_dense_87_layer_call_fn_1295907O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "ÿÿÿÿÿÿÿÿÿ/¥
E__inference_dense_88_layer_call_and_return_conditional_losses_1295937\!"/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_88_layer_call_fn_1295927O!"/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_flatten_23_layer_call_and_return_conditional_losses_1295898\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ/
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ/
 
,__inference_flatten_23_layer_call_fn_1295892O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ/
ª "ÿÿÿÿÿÿÿÿÿ/§
%__inference_signature_wrapper_1295823~!"A¢>
¢ 
7ª4
2
input_32&#
input_32ÿÿÿÿÿÿÿÿÿ/"3ª0
.
dense_88"
dense_88ÿÿÿÿÿÿÿÿÿ