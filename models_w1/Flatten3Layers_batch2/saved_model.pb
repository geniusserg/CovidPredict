¤Û
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
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8ÙÎ

Adam/dense_71/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_71/bias/v
y
(Adam/dense_71/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_71/bias/v*
_output_shapes
:*
dtype0

Adam/dense_71/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_71/kernel/v

*Adam/dense_71/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_71/kernel/v*
_output_shapes

: *
dtype0

Adam/dense_70/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_70/bias/v
y
(Adam/dense_70/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_70/bias/v*
_output_shapes
: *
dtype0

Adam/dense_70/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_70/kernel/v

*Adam/dense_70/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_70/kernel/v*
_output_shapes

: *
dtype0

Adam/dense_69/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_69/bias/v
y
(Adam/dense_69/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_69/bias/v*
_output_shapes
:*
dtype0

Adam/dense_69/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_69/kernel/v

*Adam/dense_69/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_69/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_71/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_71/bias/m
y
(Adam/dense_71/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_71/bias/m*
_output_shapes
:*
dtype0

Adam/dense_71/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_71/kernel/m

*Adam/dense_71/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_71/kernel/m*
_output_shapes

: *
dtype0

Adam/dense_70/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_70/bias/m
y
(Adam/dense_70/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_70/bias/m*
_output_shapes
: *
dtype0

Adam/dense_70/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_70/kernel/m

*Adam/dense_70/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_70/kernel/m*
_output_shapes

: *
dtype0

Adam/dense_69/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_69/bias/m
y
(Adam/dense_69/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_69/bias/m*
_output_shapes
:*
dtype0

Adam/dense_69/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_69/kernel/m

*Adam/dense_69/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_69/kernel/m*
_output_shapes

:*
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
dense_71/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_71/bias
k
!dense_71/bias/Read/ReadVariableOpReadVariableOpdense_71/bias*
_output_shapes
:*
dtype0
z
dense_71/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_71/kernel
s
#dense_71/kernel/Read/ReadVariableOpReadVariableOpdense_71/kernel*
_output_shapes

: *
dtype0
r
dense_70/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_70/bias
k
!dense_70/bias/Read/ReadVariableOpReadVariableOpdense_70/bias*
_output_shapes
: *
dtype0
z
dense_70/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_70/kernel
s
#dense_70/kernel/Read/ReadVariableOpReadVariableOpdense_70/kernel*
_output_shapes

: *
dtype0
r
dense_69/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_69/bias
k
!dense_69/bias/Read/ReadVariableOpReadVariableOpdense_69/bias*
_output_shapes
:*
dtype0
z
dense_69/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_69/kernel
s
#dense_69/kernel/Read/ReadVariableOpReadVariableOpdense_69/kernel*
_output_shapes

:*
dtype0
q
serving_default_input_28Placeholder*"
_output_shapes
:*
dtype0*
shape:

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_28dense_69/kerneldense_69/biasdense_70/kerneldense_70/biasdense_71/kerneldense_71/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_976428

NoOpNoOp
Ñ0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*0
value0Bÿ/ Bø/
Î
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¦
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias*
¦
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias*
.
0
1
"2
#3
*4
+5*
.
0
1
"2
#3
*4
+5*
* 
°
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
1trace_0
2trace_1
3trace_2
4trace_3* 
6
5trace_0
6trace_1
7trace_2
8trace_3* 
* 
°
9iter

:beta_1

;beta_2
	<decay
=learning_ratem`ma"mb#mc*md+mevfvg"vh#vi*vj+vk*

>serving_default* 
* 
* 
* 

?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Dtrace_0* 

Etrace_0* 

0
1*

0
1*
* 

Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ktrace_0* 

Ltrace_0* 
_Y
VARIABLE_VALUEdense_69/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_69/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

"0
#1*

"0
#1*
* 

Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

Rtrace_0* 

Strace_0* 
_Y
VARIABLE_VALUEdense_70/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_70/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

*0
+1*

*0
+1*
* 

Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

Ytrace_0* 

Ztrace_0* 
_Y
VARIABLE_VALUEdense_71/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_71/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

[0*
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
* 
* 
* 
* 
* 
* 
* 
8
\	variables
]	keras_api
	^total
	_count*

^0
_1*

\	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_69/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_69/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_70/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_70/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_71/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_71/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_69/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_69/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_70/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_70/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_71/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_71/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ï	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_69/kernel/Read/ReadVariableOp!dense_69/bias/Read/ReadVariableOp#dense_70/kernel/Read/ReadVariableOp!dense_70/bias/Read/ReadVariableOp#dense_71/kernel/Read/ReadVariableOp!dense_71/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_69/kernel/m/Read/ReadVariableOp(Adam/dense_69/bias/m/Read/ReadVariableOp*Adam/dense_70/kernel/m/Read/ReadVariableOp(Adam/dense_70/bias/m/Read/ReadVariableOp*Adam/dense_71/kernel/m/Read/ReadVariableOp(Adam/dense_71/bias/m/Read/ReadVariableOp*Adam/dense_69/kernel/v/Read/ReadVariableOp(Adam/dense_69/bias/v/Read/ReadVariableOp*Adam/dense_70/kernel/v/Read/ReadVariableOp(Adam/dense_70/bias/v/Read/ReadVariableOp*Adam/dense_71/kernel/v/Read/ReadVariableOp(Adam/dense_71/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_976682
ö
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_69/kerneldense_69/biasdense_70/kerneldense_70/biasdense_71/kerneldense_71/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_69/kernel/mAdam/dense_69/bias/mAdam/dense_70/kernel/mAdam/dense_70/bias/mAdam/dense_71/kernel/mAdam/dense_71/bias/mAdam/dense_69/kernel/vAdam/dense_69/bias/vAdam/dense_70/kernel/vAdam/dense_70/bias/vAdam/dense_71/kernel/vAdam/dense_71/bias/v*%
Tin
2*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_976767ÉÚ


)__inference_dense_71_layer_call_fn_976574

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallÐ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_71_layer_call_and_return_conditional_losses_976234f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

: 
 
_user_specified_nameinputs
Ñ

Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976488

inputs9
'dense_69_matmul_readvariableop_resource:6
(dense_69_biasadd_readvariableop_resource:9
'dense_70_matmul_readvariableop_resource: 6
(dense_70_biasadd_readvariableop_resource: 9
'dense_71_matmul_readvariableop_resource: 6
(dense_71_biasadd_readvariableop_resource:
identity¢dense_69/BiasAdd/ReadVariableOp¢dense_69/MatMul/ReadVariableOp¢dense_70/BiasAdd/ReadVariableOp¢dense_70/MatMul/ReadVariableOp¢dense_71/BiasAdd/ReadVariableOp¢dense_71/MatMul/ReadVariableOpa
flatten_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   i
flatten_19/ReshapeReshapeinputsflatten_19/Const:output:0*
T0*
_output_shapes

:
dense_69/MatMul/ReadVariableOpReadVariableOp'dense_69_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_69/MatMulMatMulflatten_19/Reshape:output:0&dense_69/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_69/BiasAdd/ReadVariableOpReadVariableOp(dense_69_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_69/BiasAddBiasAdddense_69/MatMul:product:0'dense_69/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Y
dense_69/ReluReludense_69/BiasAdd:output:0*
T0*
_output_shapes

:
dense_70/MatMul/ReadVariableOpReadVariableOp'dense_70_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_70/MatMulMatMuldense_69/Relu:activations:0&dense_70/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_70/BiasAdd/ReadVariableOpReadVariableOp(dense_70_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_70/BiasAddBiasAdddense_70/MatMul:product:0'dense_70/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: Y
dense_70/ReluReludense_70/BiasAdd:output:0*
T0*
_output_shapes

: 
dense_71/MatMul/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_71/MatMulMatMuldense_70/Relu:activations:0&dense_71/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_71/BiasAdd/ReadVariableOpReadVariableOp(dense_71_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_71/BiasAddBiasAdddense_71/MatMul:product:0'dense_71/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_
IdentityIdentitydense_71/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp ^dense_69/BiasAdd/ReadVariableOp^dense_69/MatMul/ReadVariableOp ^dense_70/BiasAdd/ReadVariableOp^dense_70/MatMul/ReadVariableOp ^dense_71/BiasAdd/ReadVariableOp^dense_71/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 2B
dense_69/BiasAdd/ReadVariableOpdense_69/BiasAdd/ReadVariableOp2@
dense_69/MatMul/ReadVariableOpdense_69/MatMul/ReadVariableOp2B
dense_70/BiasAdd/ReadVariableOpdense_70/BiasAdd/ReadVariableOp2@
dense_70/MatMul/ReadVariableOpdense_70/MatMul/ReadVariableOp2B
dense_71/BiasAdd/ReadVariableOpdense_71/BiasAdd/ReadVariableOp2@
dense_71/MatMul/ReadVariableOpdense_71/MatMul/ReadVariableOp:J F
"
_output_shapes
:
 
_user_specified_nameinputs

G
+__inference_flatten_19_layer_call_fn_976519

inputs
identity¨
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_19_layer_call_and_return_conditional_losses_976188W
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
å	
õ
D__inference_dense_70_layer_call_and_return_conditional_losses_976218

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

: X
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

: w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
è

6__inference_Flatten3Layers_batch2_layer_call_fn_976445

inputs
unknown:
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976241f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
î

6__inference_Flatten3Layers_batch2_layer_call_fn_976256
input_28
unknown:
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_28unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976241f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
"
_output_shapes
:
"
_user_specified_name
input_28
9
º

__inference__traced_save_976682
file_prefix.
*savev2_dense_69_kernel_read_readvariableop,
(savev2_dense_69_bias_read_readvariableop.
*savev2_dense_70_kernel_read_readvariableop,
(savev2_dense_70_bias_read_readvariableop.
*savev2_dense_71_kernel_read_readvariableop,
(savev2_dense_71_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_69_kernel_m_read_readvariableop3
/savev2_adam_dense_69_bias_m_read_readvariableop5
1savev2_adam_dense_70_kernel_m_read_readvariableop3
/savev2_adam_dense_70_bias_m_read_readvariableop5
1savev2_adam_dense_71_kernel_m_read_readvariableop3
/savev2_adam_dense_71_bias_m_read_readvariableop5
1savev2_adam_dense_69_kernel_v_read_readvariableop3
/savev2_adam_dense_69_bias_v_read_readvariableop5
1savev2_adam_dense_70_kernel_v_read_readvariableop3
/savev2_adam_dense_70_bias_v_read_readvariableop5
1savev2_adam_dense_71_kernel_v_read_readvariableop3
/savev2_adam_dense_71_bias_v_read_readvariableop
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
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*²
value¨B¥B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¡
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B ³

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_69_kernel_read_readvariableop(savev2_dense_69_bias_read_readvariableop*savev2_dense_70_kernel_read_readvariableop(savev2_dense_70_bias_read_readvariableop*savev2_dense_71_kernel_read_readvariableop(savev2_dense_71_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_69_kernel_m_read_readvariableop/savev2_adam_dense_69_bias_m_read_readvariableop1savev2_adam_dense_70_kernel_m_read_readvariableop/savev2_adam_dense_70_bias_m_read_readvariableop1savev2_adam_dense_71_kernel_m_read_readvariableop/savev2_adam_dense_71_bias_m_read_readvariableop1savev2_adam_dense_69_kernel_v_read_readvariableop/savev2_adam_dense_69_bias_v_read_readvariableop1savev2_adam_dense_70_kernel_v_read_readvariableop/savev2_adam_dense_70_bias_v_read_readvariableop1savev2_adam_dense_71_kernel_v_read_readvariableop/savev2_adam_dense_71_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	
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

identity_1Identity_1:output:0*·
_input_shapes¥
¢: ::: : : :: : : : : : : ::: : : :::: : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::
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
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
å	
õ
D__inference_dense_70_layer_call_and_return_conditional_losses_976565

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

: X
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

: w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs


)__inference_dense_70_layer_call_fn_976554

inputs
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallÐ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_70_layer_call_and_return_conditional_losses_976218f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs
å	
õ
D__inference_dense_69_layer_call_and_return_conditional_losses_976545

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:X
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

:w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs

b
F__inference_flatten_19_layer_call_and_return_conditional_losses_976525

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   S
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes

:O
IdentityIdentityReshape:output:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
¸
¡
Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976403
input_28!
dense_69_976387:
dense_69_976389:!
dense_70_976392: 
dense_70_976394: !
dense_71_976397: 
dense_71_976399:
identity¢ dense_69/StatefulPartitionedCall¢ dense_70/StatefulPartitionedCall¢ dense_71/StatefulPartitionedCallµ
flatten_19/PartitionedCallPartitionedCallinput_28*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_19_layer_call_and_return_conditional_losses_976188
 dense_69/StatefulPartitionedCallStatefulPartitionedCall#flatten_19/PartitionedCall:output:0dense_69_976387dense_69_976389*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_69_layer_call_and_return_conditional_losses_976201
 dense_70/StatefulPartitionedCallStatefulPartitionedCall)dense_69/StatefulPartitionedCall:output:0dense_70_976392dense_70_976394*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_70_layer_call_and_return_conditional_losses_976218
 dense_71/StatefulPartitionedCallStatefulPartitionedCall)dense_70/StatefulPartitionedCall:output:0dense_71_976397dense_71_976399*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_71_layer_call_and_return_conditional_losses_976234o
IdentityIdentity)dense_71/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:¯
NoOpNoOp!^dense_69/StatefulPartitionedCall!^dense_70/StatefulPartitionedCall!^dense_71/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall:L H
"
_output_shapes
:
"
_user_specified_name
input_28
å	
õ
D__inference_dense_69_layer_call_and_return_conditional_losses_976201

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:X
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

:w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
²

Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976241

inputs!
dense_69_976202:
dense_69_976204:!
dense_70_976219: 
dense_70_976221: !
dense_71_976235: 
dense_71_976237:
identity¢ dense_69/StatefulPartitionedCall¢ dense_70/StatefulPartitionedCall¢ dense_71/StatefulPartitionedCall³
flatten_19/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_19_layer_call_and_return_conditional_losses_976188
 dense_69/StatefulPartitionedCallStatefulPartitionedCall#flatten_19/PartitionedCall:output:0dense_69_976202dense_69_976204*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_69_layer_call_and_return_conditional_losses_976201
 dense_70/StatefulPartitionedCallStatefulPartitionedCall)dense_69/StatefulPartitionedCall:output:0dense_70_976219dense_70_976221*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_70_layer_call_and_return_conditional_losses_976218
 dense_71/StatefulPartitionedCallStatefulPartitionedCall)dense_70/StatefulPartitionedCall:output:0dense_71_976235dense_71_976237*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_71_layer_call_and_return_conditional_losses_976234o
IdentityIdentity)dense_71/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:¯
NoOpNoOp!^dense_69/StatefulPartitionedCall!^dense_70/StatefulPartitionedCall!^dense_71/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
²

Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976331

inputs!
dense_69_976315:
dense_69_976317:!
dense_70_976320: 
dense_70_976322: !
dense_71_976325: 
dense_71_976327:
identity¢ dense_69/StatefulPartitionedCall¢ dense_70/StatefulPartitionedCall¢ dense_71/StatefulPartitionedCall³
flatten_19/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_19_layer_call_and_return_conditional_losses_976188
 dense_69/StatefulPartitionedCallStatefulPartitionedCall#flatten_19/PartitionedCall:output:0dense_69_976315dense_69_976317*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_69_layer_call_and_return_conditional_losses_976201
 dense_70/StatefulPartitionedCallStatefulPartitionedCall)dense_69/StatefulPartitionedCall:output:0dense_70_976320dense_70_976322*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_70_layer_call_and_return_conditional_losses_976218
 dense_71/StatefulPartitionedCallStatefulPartitionedCall)dense_70/StatefulPartitionedCall:output:0dense_71_976325dense_71_976327*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_71_layer_call_and_return_conditional_losses_976234o
IdentityIdentity)dense_71/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:¯
NoOpNoOp!^dense_69/StatefulPartitionedCall!^dense_70/StatefulPartitionedCall!^dense_71/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
	
õ
D__inference_dense_71_layer_call_and_return_conditional_losses_976584

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:V
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

:w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

: 
 
_user_specified_nameinputs
è

6__inference_Flatten3Layers_batch2_layer_call_fn_976462

inputs
unknown:
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976331f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
î

6__inference_Flatten3Layers_batch2_layer_call_fn_976363
input_28
unknown:
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_28unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976331f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
"
_output_shapes
:
"
_user_specified_name
input_28
Ñ

Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976514

inputs9
'dense_69_matmul_readvariableop_resource:6
(dense_69_biasadd_readvariableop_resource:9
'dense_70_matmul_readvariableop_resource: 6
(dense_70_biasadd_readvariableop_resource: 9
'dense_71_matmul_readvariableop_resource: 6
(dense_71_biasadd_readvariableop_resource:
identity¢dense_69/BiasAdd/ReadVariableOp¢dense_69/MatMul/ReadVariableOp¢dense_70/BiasAdd/ReadVariableOp¢dense_70/MatMul/ReadVariableOp¢dense_71/BiasAdd/ReadVariableOp¢dense_71/MatMul/ReadVariableOpa
flatten_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   i
flatten_19/ReshapeReshapeinputsflatten_19/Const:output:0*
T0*
_output_shapes

:
dense_69/MatMul/ReadVariableOpReadVariableOp'dense_69_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_69/MatMulMatMulflatten_19/Reshape:output:0&dense_69/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_69/BiasAdd/ReadVariableOpReadVariableOp(dense_69_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_69/BiasAddBiasAdddense_69/MatMul:product:0'dense_69/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Y
dense_69/ReluReludense_69/BiasAdd:output:0*
T0*
_output_shapes

:
dense_70/MatMul/ReadVariableOpReadVariableOp'dense_70_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_70/MatMulMatMuldense_69/Relu:activations:0&dense_70/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_70/BiasAdd/ReadVariableOpReadVariableOp(dense_70_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_70/BiasAddBiasAdddense_70/MatMul:product:0'dense_70/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: Y
dense_70/ReluReludense_70/BiasAdd:output:0*
T0*
_output_shapes

: 
dense_71/MatMul/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_71/MatMulMatMuldense_70/Relu:activations:0&dense_71/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_71/BiasAdd/ReadVariableOpReadVariableOp(dense_71_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_71/BiasAddBiasAdddense_71/MatMul:product:0'dense_71/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_
IdentityIdentitydense_71/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp ^dense_69/BiasAdd/ReadVariableOp^dense_69/MatMul/ReadVariableOp ^dense_70/BiasAdd/ReadVariableOp^dense_70/MatMul/ReadVariableOp ^dense_71/BiasAdd/ReadVariableOp^dense_71/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 2B
dense_69/BiasAdd/ReadVariableOpdense_69/BiasAdd/ReadVariableOp2@
dense_69/MatMul/ReadVariableOpdense_69/MatMul/ReadVariableOp2B
dense_70/BiasAdd/ReadVariableOpdense_70/BiasAdd/ReadVariableOp2@
dense_70/MatMul/ReadVariableOpdense_70/MatMul/ReadVariableOp2B
dense_71/BiasAdd/ReadVariableOpdense_71/BiasAdd/ReadVariableOp2@
dense_71/MatMul/ReadVariableOpdense_71/MatMul/ReadVariableOp:J F
"
_output_shapes
:
 
_user_specified_nameinputs
¬
ÿ
$__inference_signature_wrapper_976428
input_28
unknown:
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4:
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinput_28unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_976175f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
"
_output_shapes
:
"
_user_specified_name
input_28

b
F__inference_flatten_19_layer_call_and_return_conditional_losses_976188

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   S
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes

:O
IdentityIdentityReshape:output:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
	
õ
D__inference_dense_71_layer_call_and_return_conditional_losses_976234

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:V
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

:w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

: 
 
_user_specified_nameinputs
Àe
ø
"__inference__traced_restore_976767
file_prefix2
 assignvariableop_dense_69_kernel:.
 assignvariableop_1_dense_69_bias:4
"assignvariableop_2_dense_70_kernel: .
 assignvariableop_3_dense_70_bias: 4
"assignvariableop_4_dense_71_kernel: .
 assignvariableop_5_dense_71_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: <
*assignvariableop_13_adam_dense_69_kernel_m:6
(assignvariableop_14_adam_dense_69_bias_m:<
*assignvariableop_15_adam_dense_70_kernel_m: 6
(assignvariableop_16_adam_dense_70_bias_m: <
*assignvariableop_17_adam_dense_71_kernel_m: 6
(assignvariableop_18_adam_dense_71_bias_m:<
*assignvariableop_19_adam_dense_69_kernel_v:6
(assignvariableop_20_adam_dense_69_bias_v:<
*assignvariableop_21_adam_dense_70_kernel_v: 6
(assignvariableop_22_adam_dense_70_bias_v: <
*assignvariableop_23_adam_dense_71_kernel_v: 6
(assignvariableop_24_adam_dense_71_bias_v:
identity_26¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*²
value¨B¥B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¤
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B  
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_dense_69_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_69_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_70_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_70_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_71_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_71_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_dense_69_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp(assignvariableop_14_adam_dense_69_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_70_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_70_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_71_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_71_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_69_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_69_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_70_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_70_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_71_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_71_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 õ
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_26IdentityIdentity_25:output:0^NoOp_1*
T0*
_output_shapes
: â
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_26Identity_26:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
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
¸
¡
Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976383
input_28!
dense_69_976367:
dense_69_976369:!
dense_70_976372: 
dense_70_976374: !
dense_71_976377: 
dense_71_976379:
identity¢ dense_69/StatefulPartitionedCall¢ dense_70/StatefulPartitionedCall¢ dense_71/StatefulPartitionedCallµ
flatten_19/PartitionedCallPartitionedCallinput_28*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_19_layer_call_and_return_conditional_losses_976188
 dense_69/StatefulPartitionedCallStatefulPartitionedCall#flatten_19/PartitionedCall:output:0dense_69_976367dense_69_976369*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_69_layer_call_and_return_conditional_losses_976201
 dense_70/StatefulPartitionedCallStatefulPartitionedCall)dense_69/StatefulPartitionedCall:output:0dense_70_976372dense_70_976374*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_70_layer_call_and_return_conditional_losses_976218
 dense_71/StatefulPartitionedCallStatefulPartitionedCall)dense_70/StatefulPartitionedCall:output:0dense_71_976377dense_71_976379*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_71_layer_call_and_return_conditional_losses_976234o
IdentityIdentity)dense_71/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:¯
NoOpNoOp!^dense_69/StatefulPartitionedCall!^dense_70/StatefulPartitionedCall!^dense_71/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall:L H
"
_output_shapes
:
"
_user_specified_name
input_28
(
ì
!__inference__wrapped_model_976175
input_28O
=flatten3layers_batch2_dense_69_matmul_readvariableop_resource:L
>flatten3layers_batch2_dense_69_biasadd_readvariableop_resource:O
=flatten3layers_batch2_dense_70_matmul_readvariableop_resource: L
>flatten3layers_batch2_dense_70_biasadd_readvariableop_resource: O
=flatten3layers_batch2_dense_71_matmul_readvariableop_resource: L
>flatten3layers_batch2_dense_71_biasadd_readvariableop_resource:
identity¢5Flatten3Layers_batch2/dense_69/BiasAdd/ReadVariableOp¢4Flatten3Layers_batch2/dense_69/MatMul/ReadVariableOp¢5Flatten3Layers_batch2/dense_70/BiasAdd/ReadVariableOp¢4Flatten3Layers_batch2/dense_70/MatMul/ReadVariableOp¢5Flatten3Layers_batch2/dense_71/BiasAdd/ReadVariableOp¢4Flatten3Layers_batch2/dense_71/MatMul/ReadVariableOpw
&Flatten3Layers_batch2/flatten_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
(Flatten3Layers_batch2/flatten_19/ReshapeReshapeinput_28/Flatten3Layers_batch2/flatten_19/Const:output:0*
T0*
_output_shapes

:²
4Flatten3Layers_batch2/dense_69/MatMul/ReadVariableOpReadVariableOp=flatten3layers_batch2_dense_69_matmul_readvariableop_resource*
_output_shapes

:*
dtype0É
%Flatten3Layers_batch2/dense_69/MatMulMatMul1Flatten3Layers_batch2/flatten_19/Reshape:output:0<Flatten3Layers_batch2/dense_69/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:°
5Flatten3Layers_batch2/dense_69/BiasAdd/ReadVariableOpReadVariableOp>flatten3layers_batch2_dense_69_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ê
&Flatten3Layers_batch2/dense_69/BiasAddBiasAdd/Flatten3Layers_batch2/dense_69/MatMul:product:0=Flatten3Layers_batch2/dense_69/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
#Flatten3Layers_batch2/dense_69/ReluRelu/Flatten3Layers_batch2/dense_69/BiasAdd:output:0*
T0*
_output_shapes

:²
4Flatten3Layers_batch2/dense_70/MatMul/ReadVariableOpReadVariableOp=flatten3layers_batch2_dense_70_matmul_readvariableop_resource*
_output_shapes

: *
dtype0É
%Flatten3Layers_batch2/dense_70/MatMulMatMul1Flatten3Layers_batch2/dense_69/Relu:activations:0<Flatten3Layers_batch2/dense_70/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: °
5Flatten3Layers_batch2/dense_70/BiasAdd/ReadVariableOpReadVariableOp>flatten3layers_batch2_dense_70_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ê
&Flatten3Layers_batch2/dense_70/BiasAddBiasAdd/Flatten3Layers_batch2/dense_70/MatMul:product:0=Flatten3Layers_batch2/dense_70/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
#Flatten3Layers_batch2/dense_70/ReluRelu/Flatten3Layers_batch2/dense_70/BiasAdd:output:0*
T0*
_output_shapes

: ²
4Flatten3Layers_batch2/dense_71/MatMul/ReadVariableOpReadVariableOp=flatten3layers_batch2_dense_71_matmul_readvariableop_resource*
_output_shapes

: *
dtype0É
%Flatten3Layers_batch2/dense_71/MatMulMatMul1Flatten3Layers_batch2/dense_70/Relu:activations:0<Flatten3Layers_batch2/dense_71/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:°
5Flatten3Layers_batch2/dense_71/BiasAdd/ReadVariableOpReadVariableOp>flatten3layers_batch2_dense_71_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ê
&Flatten3Layers_batch2/dense_71/BiasAddBiasAdd/Flatten3Layers_batch2/dense_71/MatMul:product:0=Flatten3Layers_batch2/dense_71/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:u
IdentityIdentity/Flatten3Layers_batch2/dense_71/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp6^Flatten3Layers_batch2/dense_69/BiasAdd/ReadVariableOp5^Flatten3Layers_batch2/dense_69/MatMul/ReadVariableOp6^Flatten3Layers_batch2/dense_70/BiasAdd/ReadVariableOp5^Flatten3Layers_batch2/dense_70/MatMul/ReadVariableOp6^Flatten3Layers_batch2/dense_71/BiasAdd/ReadVariableOp5^Flatten3Layers_batch2/dense_71/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 2n
5Flatten3Layers_batch2/dense_69/BiasAdd/ReadVariableOp5Flatten3Layers_batch2/dense_69/BiasAdd/ReadVariableOp2l
4Flatten3Layers_batch2/dense_69/MatMul/ReadVariableOp4Flatten3Layers_batch2/dense_69/MatMul/ReadVariableOp2n
5Flatten3Layers_batch2/dense_70/BiasAdd/ReadVariableOp5Flatten3Layers_batch2/dense_70/BiasAdd/ReadVariableOp2l
4Flatten3Layers_batch2/dense_70/MatMul/ReadVariableOp4Flatten3Layers_batch2/dense_70/MatMul/ReadVariableOp2n
5Flatten3Layers_batch2/dense_71/BiasAdd/ReadVariableOp5Flatten3Layers_batch2/dense_71/BiasAdd/ReadVariableOp2l
4Flatten3Layers_batch2/dense_71/MatMul/ReadVariableOp4Flatten3Layers_batch2/dense_71/MatMul/ReadVariableOp:L H
"
_output_shapes
:
"
_user_specified_name
input_28


)__inference_dense_69_layer_call_fn_976534

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÐ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_69_layer_call_and_return_conditional_losses_976201f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
8
input_28,
serving_default_input_28:03
dense_71'
StatefulPartitionedCall:0tensorflow/serving/predict:{
è
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias"
_tf_keras_layer
»
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias"
_tf_keras_layer
J
0
1
"2
#3
*4
+5"
trackable_list_wrapper
J
0
1
"2
#3
*4
+5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object

1trace_0
2trace_1
3trace_2
4trace_32¢
6__inference_Flatten3Layers_batch2_layer_call_fn_976256
6__inference_Flatten3Layers_batch2_layer_call_fn_976445
6__inference_Flatten3Layers_batch2_layer_call_fn_976462
6__inference_Flatten3Layers_batch2_layer_call_fn_976363¿
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
 z1trace_0z2trace_1z3trace_2z4trace_3
ù
5trace_0
6trace_1
7trace_2
8trace_32
Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976488
Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976514
Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976383
Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976403¿
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
 z5trace_0z6trace_1z7trace_2z8trace_3
ÍBÊ
!__inference__wrapped_model_976175input_28"
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
¿
9iter

:beta_1

;beta_2
	<decay
=learning_ratem`ma"mb#mc*md+mevfvg"vh#vi*vj+vk"
	optimizer
,
>serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ï
Dtrace_02Ò
+__inference_flatten_19_layer_call_fn_976519¢
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
 zDtrace_0

Etrace_02í
F__inference_flatten_19_layer_call_and_return_conditional_losses_976525¢
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
 zEtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
í
Ktrace_02Ð
)__inference_dense_69_layer_call_fn_976534¢
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
 zKtrace_0

Ltrace_02ë
D__inference_dense_69_layer_call_and_return_conditional_losses_976545¢
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
 zLtrace_0
!:2dense_69/kernel
:2dense_69/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
í
Rtrace_02Ð
)__inference_dense_70_layer_call_fn_976554¢
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
 zRtrace_0

Strace_02ë
D__inference_dense_70_layer_call_and_return_conditional_losses_976565¢
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
 zStrace_0
!: 2dense_70/kernel
: 2dense_70/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
í
Ytrace_02Ð
)__inference_dense_71_layer_call_fn_976574¢
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
 zYtrace_0

Ztrace_02ë
D__inference_dense_71_layer_call_and_return_conditional_losses_976584¢
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
 zZtrace_0
!: 2dense_71/kernel
:2dense_71/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
[0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
6__inference_Flatten3Layers_batch2_layer_call_fn_976256input_28"¿
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
6__inference_Flatten3Layers_batch2_layer_call_fn_976445inputs"¿
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
6__inference_Flatten3Layers_batch2_layer_call_fn_976462inputs"¿
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
6__inference_Flatten3Layers_batch2_layer_call_fn_976363input_28"¿
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
Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976488inputs"¿
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
Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976514inputs"¿
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
Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976383input_28"¿
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
Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976403input_28"¿
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
ÌBÉ
$__inference_signature_wrapper_976428input_28"
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
ßBÜ
+__inference_flatten_19_layer_call_fn_976519inputs"¢
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
úB÷
F__inference_flatten_19_layer_call_and_return_conditional_losses_976525inputs"¢
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
ÝBÚ
)__inference_dense_69_layer_call_fn_976534inputs"¢
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
øBõ
D__inference_dense_69_layer_call_and_return_conditional_losses_976545inputs"¢
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
ÝBÚ
)__inference_dense_70_layer_call_fn_976554inputs"¢
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
øBõ
D__inference_dense_70_layer_call_and_return_conditional_losses_976565inputs"¢
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
ÝBÚ
)__inference_dense_71_layer_call_fn_976574inputs"¢
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
øBõ
D__inference_dense_71_layer_call_and_return_conditional_losses_976584inputs"¢
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
\	variables
]	keras_api
	^total
	_count"
_tf_keras_metric
.
^0
_1"
trackable_list_wrapper
-
\	variables"
_generic_user_object
:  (2total
:  (2count
&:$2Adam/dense_69/kernel/m
 :2Adam/dense_69/bias/m
&:$ 2Adam/dense_70/kernel/m
 : 2Adam/dense_70/bias/m
&:$ 2Adam/dense_71/kernel/m
 :2Adam/dense_71/bias/m
&:$2Adam/dense_69/kernel/v
 :2Adam/dense_69/bias/v
&:$ 2Adam/dense_70/kernel/v
 : 2Adam/dense_70/bias/v
&:$ 2Adam/dense_71/kernel/v
 :2Adam/dense_71/bias/v±
Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976383\"#*+4¢1
*¢'

input_28
p 

 
ª "¢

0
 ±
Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976403\"#*+4¢1
*¢'

input_28
p

 
ª "¢

0
 ¯
Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976488Z"#*+2¢/
(¢%

inputs
p 

 
ª "¢

0
 ¯
Q__inference_Flatten3Layers_batch2_layer_call_and_return_conditional_losses_976514Z"#*+2¢/
(¢%

inputs
p

 
ª "¢

0
 
6__inference_Flatten3Layers_batch2_layer_call_fn_976256O"#*+4¢1
*¢'

input_28
p 

 
ª "
6__inference_Flatten3Layers_batch2_layer_call_fn_976363O"#*+4¢1
*¢'

input_28
p

 
ª "
6__inference_Flatten3Layers_batch2_layer_call_fn_976445M"#*+2¢/
(¢%

inputs
p 

 
ª "
6__inference_Flatten3Layers_batch2_layer_call_fn_976462M"#*+2¢/
(¢%

inputs
p

 
ª "
!__inference__wrapped_model_976175b"#*+,¢)
"¢

input_28
ª "*ª'
%
dense_71
dense_71
D__inference_dense_69_layer_call_and_return_conditional_losses_976545J&¢#
¢

inputs
ª "¢

0
 j
)__inference_dense_69_layer_call_fn_976534=&¢#
¢

inputs
ª "
D__inference_dense_70_layer_call_and_return_conditional_losses_976565J"#&¢#
¢

inputs
ª "¢

0 
 j
)__inference_dense_70_layer_call_fn_976554="#&¢#
¢

inputs
ª " 
D__inference_dense_71_layer_call_and_return_conditional_losses_976584J*+&¢#
¢

inputs 
ª "¢

0
 j
)__inference_dense_71_layer_call_fn_976574=*+&¢#
¢

inputs 
ª "
F__inference_flatten_19_layer_call_and_return_conditional_losses_976525J*¢'
 ¢

inputs
ª "¢

0
 l
+__inference_flatten_19_layer_call_fn_976519=*¢'
 ¢

inputs
ª "
$__inference_signature_wrapper_976428n"#*+8¢5
¢ 
.ª+
)
input_28
input_28"*ª'
%
dense_71
dense_71