þ
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
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8×Û

Adam/dense_95/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_95/bias/v
y
(Adam/dense_95/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_95/bias/v*
_output_shapes
:*
dtype0

Adam/dense_95/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_95/kernel/v

*Adam/dense_95/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_95/kernel/v*
_output_shapes

: *
dtype0

Adam/dense_94/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_94/bias/v
y
(Adam/dense_94/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_94/bias/v*
_output_shapes
: *
dtype0

Adam/dense_94/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_94/kernel/v

*Adam/dense_94/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_94/kernel/v*
_output_shapes

:  *
dtype0

Adam/dense_93/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_93/bias/v
y
(Adam/dense_93/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_93/bias/v*
_output_shapes
: *
dtype0

Adam/dense_93/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/ *'
shared_nameAdam/dense_93/kernel/v

*Adam/dense_93/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_93/kernel/v*
_output_shapes

:/ *
dtype0

Adam/dense_92/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_92/bias/v
y
(Adam/dense_92/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_92/bias/v*
_output_shapes
:/*
dtype0

Adam/dense_92/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*'
shared_nameAdam/dense_92/kernel/v

*Adam/dense_92/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_92/kernel/v*
_output_shapes

://*
dtype0

Adam/dense_95/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_95/bias/m
y
(Adam/dense_95/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_95/bias/m*
_output_shapes
:*
dtype0

Adam/dense_95/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_95/kernel/m

*Adam/dense_95/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_95/kernel/m*
_output_shapes

: *
dtype0

Adam/dense_94/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_94/bias/m
y
(Adam/dense_94/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_94/bias/m*
_output_shapes
: *
dtype0

Adam/dense_94/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_94/kernel/m

*Adam/dense_94/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_94/kernel/m*
_output_shapes

:  *
dtype0

Adam/dense_93/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_93/bias/m
y
(Adam/dense_93/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_93/bias/m*
_output_shapes
: *
dtype0

Adam/dense_93/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/ *'
shared_nameAdam/dense_93/kernel/m

*Adam/dense_93/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_93/kernel/m*
_output_shapes

:/ *
dtype0

Adam/dense_92/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_92/bias/m
y
(Adam/dense_92/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_92/bias/m*
_output_shapes
:/*
dtype0

Adam/dense_92/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*'
shared_nameAdam/dense_92/kernel/m

*Adam/dense_92/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_92/kernel/m*
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
dense_95/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_95/bias
k
!dense_95/bias/Read/ReadVariableOpReadVariableOpdense_95/bias*
_output_shapes
:*
dtype0
z
dense_95/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_95/kernel
s
#dense_95/kernel/Read/ReadVariableOpReadVariableOpdense_95/kernel*
_output_shapes

: *
dtype0
r
dense_94/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_94/bias
k
!dense_94/bias/Read/ReadVariableOpReadVariableOpdense_94/bias*
_output_shapes
: *
dtype0
z
dense_94/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_94/kernel
s
#dense_94/kernel/Read/ReadVariableOpReadVariableOpdense_94/kernel*
_output_shapes

:  *
dtype0
r
dense_93/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_93/bias
k
!dense_93/bias/Read/ReadVariableOpReadVariableOpdense_93/bias*
_output_shapes
: *
dtype0
z
dense_93/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/ * 
shared_namedense_93/kernel
s
#dense_93/kernel/Read/ReadVariableOpReadVariableOpdense_93/kernel*
_output_shapes

:/ *
dtype0
r
dense_92/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namedense_92/bias
k
!dense_92/bias/Read/ReadVariableOpReadVariableOpdense_92/bias*
_output_shapes
:/*
dtype0
z
dense_92/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
://* 
shared_namedense_92/kernel
s
#dense_92/kernel/Read/ReadVariableOpReadVariableOpdense_92/kernel*
_output_shapes

://*
dtype0

serving_default_input_34Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ/
Ä
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_34dense_92/kerneldense_92/biasdense_93/kerneldense_93/biasdense_94/kerneldense_94/biasdense_95/kerneldense_95/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1321145

NoOpNoOp
°;
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ë:
valueá:BÞ: B×:
õ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¦
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias*
¦
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias*
¦
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias*
<
0
1
#2
$3
+4
,5
36
47*
<
0
1
#2
$3
+4
,5
36
47*
* 
°
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
:trace_0
;trace_1
<trace_2
=trace_3* 
6
>trace_0
?trace_1
@trace_2
Atrace_3* 
* 
Ô
Biter

Cbeta_1

Dbeta_2
	Edecay
Flearning_ratempmq#mr$ms+mt,mu3mv4mwvxvy#vz$v{+v|,v}3v~4v*

Gserving_default* 
* 
* 
* 

Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Mtrace_0* 

Ntrace_0* 

0
1*

0
1*
* 

Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ttrace_0* 

Utrace_0* 
_Y
VARIABLE_VALUEdense_92/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_92/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

#0
$1*

#0
$1*
* 

Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

[trace_0* 

\trace_0* 
_Y
VARIABLE_VALUEdense_93/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_93/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

+0
,1*

+0
,1*
* 

]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

btrace_0* 

ctrace_0* 
_Y
VARIABLE_VALUEdense_94/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_94/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

30
41*

30
41*
* 

dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

itrace_0* 

jtrace_0* 
_Y
VARIABLE_VALUEdense_95/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_95/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

k0*
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
* 
* 
* 
* 
* 
* 
* 
8
l	variables
m	keras_api
	ntotal
	ocount*

n0
o1*

l	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_92/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_92/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_93/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_93/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_94/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_94/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_95/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_95/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_92/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_92/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_93/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_93/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_94/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_94/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_95/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_95/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ê
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_92/kernel/Read/ReadVariableOp!dense_92/bias/Read/ReadVariableOp#dense_93/kernel/Read/ReadVariableOp!dense_93/bias/Read/ReadVariableOp#dense_94/kernel/Read/ReadVariableOp!dense_94/bias/Read/ReadVariableOp#dense_95/kernel/Read/ReadVariableOp!dense_95/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_92/kernel/m/Read/ReadVariableOp(Adam/dense_92/bias/m/Read/ReadVariableOp*Adam/dense_93/kernel/m/Read/ReadVariableOp(Adam/dense_93/bias/m/Read/ReadVariableOp*Adam/dense_94/kernel/m/Read/ReadVariableOp(Adam/dense_94/bias/m/Read/ReadVariableOp*Adam/dense_95/kernel/m/Read/ReadVariableOp(Adam/dense_95/bias/m/Read/ReadVariableOp*Adam/dense_92/kernel/v/Read/ReadVariableOp(Adam/dense_92/bias/v/Read/ReadVariableOp*Adam/dense_93/kernel/v/Read/ReadVariableOp(Adam/dense_93/bias/v/Read/ReadVariableOp*Adam/dense_94/kernel/v/Read/ReadVariableOp(Adam/dense_94/bias/v/Read/ReadVariableOp*Adam/dense_95/kernel/v/Read/ReadVariableOp(Adam/dense_95/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
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
 __inference__traced_save_1321453
ù
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_92/kerneldense_92/biasdense_93/kerneldense_93/biasdense_94/kerneldense_94/biasdense_95/kerneldense_95/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_92/kernel/mAdam/dense_92/bias/mAdam/dense_93/kernel/mAdam/dense_93/bias/mAdam/dense_94/kernel/mAdam/dense_94/bias/mAdam/dense_95/kernel/mAdam/dense_95/bias/mAdam/dense_92/kernel/vAdam/dense_92/bias/vAdam/dense_93/kernel/vAdam/dense_93/bias/vAdam/dense_94/kernel/vAdam/dense_94/bias/vAdam/dense_95/kernel/vAdam/dense_95/bias/v*+
Tin$
"2 *
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
#__inference__traced_restore_1321556½Í
æ	
Ç
6__inference_Flatten4Layers_Mixed_layer_call_fn_1320932
input_34
unknown://
	unknown_0:/
	unknown_1:/ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinput_34unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1320913o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ/: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
"
_user_specified_name
input_34
È	
ö
E__inference_dense_95_layer_call_and_return_conditional_losses_1320906

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
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
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¿
c
G__inference_flatten_25_layer_call_and_return_conditional_losses_1321260

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
È	
ö
E__inference_dense_94_layer_call_and_return_conditional_losses_1320890

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
}
Ñ
#__inference__traced_restore_1321556
file_prefix2
 assignvariableop_dense_92_kernel://.
 assignvariableop_1_dense_92_bias:/4
"assignvariableop_2_dense_93_kernel:/ .
 assignvariableop_3_dense_93_bias: 4
"assignvariableop_4_dense_94_kernel:  .
 assignvariableop_5_dense_94_bias: 4
"assignvariableop_6_dense_95_kernel: .
 assignvariableop_7_dense_95_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: <
*assignvariableop_15_adam_dense_92_kernel_m://6
(assignvariableop_16_adam_dense_92_bias_m:/<
*assignvariableop_17_adam_dense_93_kernel_m:/ 6
(assignvariableop_18_adam_dense_93_bias_m: <
*assignvariableop_19_adam_dense_94_kernel_m:  6
(assignvariableop_20_adam_dense_94_bias_m: <
*assignvariableop_21_adam_dense_95_kernel_m: 6
(assignvariableop_22_adam_dense_95_bias_m:<
*assignvariableop_23_adam_dense_92_kernel_v://6
(assignvariableop_24_adam_dense_92_bias_v:/<
*assignvariableop_25_adam_dense_93_kernel_v:/ 6
(assignvariableop_26_adam_dense_93_bias_v: <
*assignvariableop_27_adam_dense_94_kernel_v:  6
(assignvariableop_28_adam_dense_94_bias_v: <
*assignvariableop_29_adam_dense_95_kernel_v: 6
(assignvariableop_30_adam_dense_95_bias_v:
identity_32¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Æ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*ì
valueâBß B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH°
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Á
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_dense_92_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_92_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_93_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_93_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_94_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_94_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_95_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_95_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_92_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_92_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_93_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_93_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_94_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_94_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_95_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_95_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_92_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_92_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_93_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_93_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_94_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_94_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_95_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_95_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ù
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: æ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
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
Ä

*__inference_dense_94_layer_call_fn_1321308

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_94_layer_call_and_return_conditional_losses_1320890o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ô

Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1321116
input_34"
dense_92_1321095://
dense_92_1321097:/"
dense_93_1321100:/ 
dense_93_1321102: "
dense_94_1321105:  
dense_94_1321107: "
dense_95_1321110: 
dense_95_1321112:
identity¢ dense_92/StatefulPartitionedCall¢ dense_93/StatefulPartitionedCall¢ dense_94/StatefulPartitionedCall¢ dense_95/StatefulPartitionedCall¿
flatten_25/PartitionedCallPartitionedCallinput_34*
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
G__inference_flatten_25_layer_call_and_return_conditional_losses_1320845
 dense_92/StatefulPartitionedCallStatefulPartitionedCall#flatten_25/PartitionedCall:output:0dense_92_1321095dense_92_1321097*
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
E__inference_dense_92_layer_call_and_return_conditional_losses_1320858
 dense_93/StatefulPartitionedCallStatefulPartitionedCall)dense_92/StatefulPartitionedCall:output:0dense_93_1321100dense_93_1321102*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_93_layer_call_and_return_conditional_losses_1320874
 dense_94/StatefulPartitionedCallStatefulPartitionedCall)dense_93/StatefulPartitionedCall:output:0dense_94_1321105dense_94_1321107*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_94_layer_call_and_return_conditional_losses_1320890
 dense_95/StatefulPartitionedCallStatefulPartitionedCall)dense_94/StatefulPartitionedCall:output:0dense_95_1321110dense_95_1321112*
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
E__inference_dense_95_layer_call_and_return_conditional_losses_1320906x
IdentityIdentity)dense_95/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
NoOpNoOp!^dense_92/StatefulPartitionedCall!^dense_93/StatefulPartitionedCall!^dense_94/StatefulPartitionedCall!^dense_95/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ/: : : : : : : : 2D
 dense_92/StatefulPartitionedCall dense_92/StatefulPartitionedCall2D
 dense_93/StatefulPartitionedCall dense_93/StatefulPartitionedCall2D
 dense_94/StatefulPartitionedCall dense_94/StatefulPartitionedCall2D
 dense_95/StatefulPartitionedCall dense_95/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
"
_user_specified_name
input_34
«
H
,__inference_flatten_25_layer_call_fn_1321254

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
G__inference_flatten_25_layer_call_and_return_conditional_losses_1320845`
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
ô

Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1321091
input_34"
dense_92_1321070://
dense_92_1321072:/"
dense_93_1321075:/ 
dense_93_1321077: "
dense_94_1321080:  
dense_94_1321082: "
dense_95_1321085: 
dense_95_1321087:
identity¢ dense_92/StatefulPartitionedCall¢ dense_93/StatefulPartitionedCall¢ dense_94/StatefulPartitionedCall¢ dense_95/StatefulPartitionedCall¿
flatten_25/PartitionedCallPartitionedCallinput_34*
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
G__inference_flatten_25_layer_call_and_return_conditional_losses_1320845
 dense_92/StatefulPartitionedCallStatefulPartitionedCall#flatten_25/PartitionedCall:output:0dense_92_1321070dense_92_1321072*
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
E__inference_dense_92_layer_call_and_return_conditional_losses_1320858
 dense_93/StatefulPartitionedCallStatefulPartitionedCall)dense_92/StatefulPartitionedCall:output:0dense_93_1321075dense_93_1321077*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_93_layer_call_and_return_conditional_losses_1320874
 dense_94/StatefulPartitionedCallStatefulPartitionedCall)dense_93/StatefulPartitionedCall:output:0dense_94_1321080dense_94_1321082*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_94_layer_call_and_return_conditional_losses_1320890
 dense_95/StatefulPartitionedCallStatefulPartitionedCall)dense_94/StatefulPartitionedCall:output:0dense_95_1321085dense_95_1321087*
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
E__inference_dense_95_layer_call_and_return_conditional_losses_1320906x
IdentityIdentity)dense_95/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
NoOpNoOp!^dense_92/StatefulPartitionedCall!^dense_93/StatefulPartitionedCall!^dense_94/StatefulPartitionedCall!^dense_95/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ/: : : : : : : : 2D
 dense_92/StatefulPartitionedCall dense_92/StatefulPartitionedCall2D
 dense_93/StatefulPartitionedCall dense_93/StatefulPartitionedCall2D
 dense_94/StatefulPartitionedCall dense_94/StatefulPartitionedCall2D
 dense_95/StatefulPartitionedCall dense_95/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
"
_user_specified_name
input_34
$
È
Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1321249

inputs9
'dense_92_matmul_readvariableop_resource://6
(dense_92_biasadd_readvariableop_resource:/9
'dense_93_matmul_readvariableop_resource:/ 6
(dense_93_biasadd_readvariableop_resource: 9
'dense_94_matmul_readvariableop_resource:  6
(dense_94_biasadd_readvariableop_resource: 9
'dense_95_matmul_readvariableop_resource: 6
(dense_95_biasadd_readvariableop_resource:
identity¢dense_92/BiasAdd/ReadVariableOp¢dense_92/MatMul/ReadVariableOp¢dense_93/BiasAdd/ReadVariableOp¢dense_93/MatMul/ReadVariableOp¢dense_94/BiasAdd/ReadVariableOp¢dense_94/MatMul/ReadVariableOp¢dense_95/BiasAdd/ReadVariableOp¢dense_95/MatMul/ReadVariableOpa
flatten_25/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ/   r
flatten_25/ReshapeReshapeinputsflatten_25/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
dense_92/MatMul/ReadVariableOpReadVariableOp'dense_92_matmul_readvariableop_resource*
_output_shapes

://*
dtype0
dense_92/MatMulMatMulflatten_25/Reshape:output:0&dense_92/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
dense_92/BiasAdd/ReadVariableOpReadVariableOp(dense_92_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0
dense_92/BiasAddBiasAdddense_92/MatMul:product:0'dense_92/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/b
dense_92/ReluReludense_92/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
dense_93/MatMul/ReadVariableOpReadVariableOp'dense_93_matmul_readvariableop_resource*
_output_shapes

:/ *
dtype0
dense_93/MatMulMatMuldense_92/Relu:activations:0&dense_93/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_93/BiasAdd/ReadVariableOpReadVariableOp(dense_93_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_93/BiasAddBiasAdddense_93/MatMul:product:0'dense_93/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_94/MatMul/ReadVariableOpReadVariableOp'dense_94_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_94/MatMulMatMuldense_93/BiasAdd:output:0&dense_94/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_94/BiasAdd/ReadVariableOpReadVariableOp(dense_94_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_94/BiasAddBiasAdddense_94/MatMul:product:0'dense_94/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_95/MatMul/ReadVariableOpReadVariableOp'dense_95_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_95/MatMulMatMuldense_94/BiasAdd:output:0&dense_95/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_95/BiasAdd/ReadVariableOpReadVariableOp(dense_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_95/BiasAddBiasAdddense_95/MatMul:product:0'dense_95/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_95/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
NoOpNoOp ^dense_92/BiasAdd/ReadVariableOp^dense_92/MatMul/ReadVariableOp ^dense_93/BiasAdd/ReadVariableOp^dense_93/MatMul/ReadVariableOp ^dense_94/BiasAdd/ReadVariableOp^dense_94/MatMul/ReadVariableOp ^dense_95/BiasAdd/ReadVariableOp^dense_95/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ/: : : : : : : : 2B
dense_92/BiasAdd/ReadVariableOpdense_92/BiasAdd/ReadVariableOp2@
dense_92/MatMul/ReadVariableOpdense_92/MatMul/ReadVariableOp2B
dense_93/BiasAdd/ReadVariableOpdense_93/BiasAdd/ReadVariableOp2@
dense_93/MatMul/ReadVariableOpdense_93/MatMul/ReadVariableOp2B
dense_94/BiasAdd/ReadVariableOpdense_94/BiasAdd/ReadVariableOp2@
dense_94/MatMul/ReadVariableOpdense_94/MatMul/ReadVariableOp2B
dense_95/BiasAdd/ReadVariableOpdense_95/BiasAdd/ReadVariableOp2@
dense_95/MatMul/ReadVariableOpdense_95/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
È	
ö
E__inference_dense_94_layer_call_and_return_conditional_losses_1321318

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
È	
ö
E__inference_dense_95_layer_call_and_return_conditional_losses_1321337

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
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
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
$
È
Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1321218

inputs9
'dense_92_matmul_readvariableop_resource://6
(dense_92_biasadd_readvariableop_resource:/9
'dense_93_matmul_readvariableop_resource:/ 6
(dense_93_biasadd_readvariableop_resource: 9
'dense_94_matmul_readvariableop_resource:  6
(dense_94_biasadd_readvariableop_resource: 9
'dense_95_matmul_readvariableop_resource: 6
(dense_95_biasadd_readvariableop_resource:
identity¢dense_92/BiasAdd/ReadVariableOp¢dense_92/MatMul/ReadVariableOp¢dense_93/BiasAdd/ReadVariableOp¢dense_93/MatMul/ReadVariableOp¢dense_94/BiasAdd/ReadVariableOp¢dense_94/MatMul/ReadVariableOp¢dense_95/BiasAdd/ReadVariableOp¢dense_95/MatMul/ReadVariableOpa
flatten_25/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ/   r
flatten_25/ReshapeReshapeinputsflatten_25/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
dense_92/MatMul/ReadVariableOpReadVariableOp'dense_92_matmul_readvariableop_resource*
_output_shapes

://*
dtype0
dense_92/MatMulMatMulflatten_25/Reshape:output:0&dense_92/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
dense_92/BiasAdd/ReadVariableOpReadVariableOp(dense_92_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0
dense_92/BiasAddBiasAdddense_92/MatMul:product:0'dense_92/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/b
dense_92/ReluReludense_92/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
dense_93/MatMul/ReadVariableOpReadVariableOp'dense_93_matmul_readvariableop_resource*
_output_shapes

:/ *
dtype0
dense_93/MatMulMatMuldense_92/Relu:activations:0&dense_93/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_93/BiasAdd/ReadVariableOpReadVariableOp(dense_93_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_93/BiasAddBiasAdddense_93/MatMul:product:0'dense_93/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_94/MatMul/ReadVariableOpReadVariableOp'dense_94_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_94/MatMulMatMuldense_93/BiasAdd:output:0&dense_94/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_94/BiasAdd/ReadVariableOpReadVariableOp(dense_94_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_94/BiasAddBiasAdddense_94/MatMul:product:0'dense_94/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_95/MatMul/ReadVariableOpReadVariableOp'dense_95_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_95/MatMulMatMuldense_94/BiasAdd:output:0&dense_95/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_95/BiasAdd/ReadVariableOpReadVariableOp(dense_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_95/BiasAddBiasAdddense_95/MatMul:product:0'dense_95/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_95/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
NoOpNoOp ^dense_92/BiasAdd/ReadVariableOp^dense_92/MatMul/ReadVariableOp ^dense_93/BiasAdd/ReadVariableOp^dense_93/MatMul/ReadVariableOp ^dense_94/BiasAdd/ReadVariableOp^dense_94/MatMul/ReadVariableOp ^dense_95/BiasAdd/ReadVariableOp^dense_95/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ/: : : : : : : : 2B
dense_92/BiasAdd/ReadVariableOpdense_92/BiasAdd/ReadVariableOp2@
dense_92/MatMul/ReadVariableOpdense_92/MatMul/ReadVariableOp2B
dense_93/BiasAdd/ReadVariableOpdense_93/BiasAdd/ReadVariableOp2@
dense_93/MatMul/ReadVariableOpdense_93/MatMul/ReadVariableOp2B
dense_94/BiasAdd/ReadVariableOpdense_94/BiasAdd/ReadVariableOp2@
dense_94/MatMul/ReadVariableOpdense_94/MatMul/ReadVariableOp2B
dense_95/BiasAdd/ReadVariableOpdense_95/BiasAdd/ReadVariableOp2@
dense_95/MatMul/ReadVariableOpdense_95/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
î

Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1321026

inputs"
dense_92_1321005://
dense_92_1321007:/"
dense_93_1321010:/ 
dense_93_1321012: "
dense_94_1321015:  
dense_94_1321017: "
dense_95_1321020: 
dense_95_1321022:
identity¢ dense_92/StatefulPartitionedCall¢ dense_93/StatefulPartitionedCall¢ dense_94/StatefulPartitionedCall¢ dense_95/StatefulPartitionedCall½
flatten_25/PartitionedCallPartitionedCallinputs*
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
G__inference_flatten_25_layer_call_and_return_conditional_losses_1320845
 dense_92/StatefulPartitionedCallStatefulPartitionedCall#flatten_25/PartitionedCall:output:0dense_92_1321005dense_92_1321007*
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
E__inference_dense_92_layer_call_and_return_conditional_losses_1320858
 dense_93/StatefulPartitionedCallStatefulPartitionedCall)dense_92/StatefulPartitionedCall:output:0dense_93_1321010dense_93_1321012*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_93_layer_call_and_return_conditional_losses_1320874
 dense_94/StatefulPartitionedCallStatefulPartitionedCall)dense_93/StatefulPartitionedCall:output:0dense_94_1321015dense_94_1321017*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_94_layer_call_and_return_conditional_losses_1320890
 dense_95/StatefulPartitionedCallStatefulPartitionedCall)dense_94/StatefulPartitionedCall:output:0dense_95_1321020dense_95_1321022*
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
E__inference_dense_95_layer_call_and_return_conditional_losses_1320906x
IdentityIdentity)dense_95/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
NoOpNoOp!^dense_92/StatefulPartitionedCall!^dense_93/StatefulPartitionedCall!^dense_94/StatefulPartitionedCall!^dense_95/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ/: : : : : : : : 2D
 dense_92/StatefulPartitionedCall dense_92/StatefulPartitionedCall2D
 dense_93/StatefulPartitionedCall dense_93/StatefulPartitionedCall2D
 dense_94/StatefulPartitionedCall dense_94/StatefulPartitionedCall2D
 dense_95/StatefulPartitionedCall dense_95/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
È	
ö
E__inference_dense_93_layer_call_and_return_conditional_losses_1321299

inputs0
matmul_readvariableop_resource:/ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
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
È	
ö
E__inference_dense_93_layer_call_and_return_conditional_losses_1320874

inputs0
matmul_readvariableop_resource:/ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
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
Ä

*__inference_dense_93_layer_call_fn_1321289

inputs
unknown:/ 
	unknown_0: 
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_93_layer_call_and_return_conditional_losses_1320874o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
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
î

Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1320913

inputs"
dense_92_1320859://
dense_92_1320861:/"
dense_93_1320875:/ 
dense_93_1320877: "
dense_94_1320891:  
dense_94_1320893: "
dense_95_1320907: 
dense_95_1320909:
identity¢ dense_92/StatefulPartitionedCall¢ dense_93/StatefulPartitionedCall¢ dense_94/StatefulPartitionedCall¢ dense_95/StatefulPartitionedCall½
flatten_25/PartitionedCallPartitionedCallinputs*
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
G__inference_flatten_25_layer_call_and_return_conditional_losses_1320845
 dense_92/StatefulPartitionedCallStatefulPartitionedCall#flatten_25/PartitionedCall:output:0dense_92_1320859dense_92_1320861*
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
E__inference_dense_92_layer_call_and_return_conditional_losses_1320858
 dense_93/StatefulPartitionedCallStatefulPartitionedCall)dense_92/StatefulPartitionedCall:output:0dense_93_1320875dense_93_1320877*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_93_layer_call_and_return_conditional_losses_1320874
 dense_94/StatefulPartitionedCallStatefulPartitionedCall)dense_93/StatefulPartitionedCall:output:0dense_94_1320891dense_94_1320893*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_94_layer_call_and_return_conditional_losses_1320890
 dense_95/StatefulPartitionedCallStatefulPartitionedCall)dense_94/StatefulPartitionedCall:output:0dense_95_1320907dense_95_1320909*
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
E__inference_dense_95_layer_call_and_return_conditional_losses_1320906x
IdentityIdentity)dense_95/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
NoOpNoOp!^dense_92/StatefulPartitionedCall!^dense_93/StatefulPartitionedCall!^dense_94/StatefulPartitionedCall!^dense_95/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ/: : : : : : : : 2D
 dense_92/StatefulPartitionedCall dense_92/StatefulPartitionedCall2D
 dense_93/StatefulPartitionedCall dense_93/StatefulPartitionedCall2D
 dense_94/StatefulPartitionedCall dense_94/StatefulPartitionedCall2D
 dense_95/StatefulPartitionedCall dense_95/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
Ä

*__inference_dense_92_layer_call_fn_1321269

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
E__inference_dense_92_layer_call_and_return_conditional_losses_1320858o
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
æ	
Ç
6__inference_Flatten4Layers_Mixed_layer_call_fn_1321066
input_34
unknown://
	unknown_0:/
	unknown_1:/ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinput_34unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1321026o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ/: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
"
_user_specified_name
input_34
Ä

*__inference_dense_95_layer_call_fn_1321327

inputs
unknown: 
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
E__inference_dense_95_layer_call_and_return_conditional_losses_1320906o
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
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


ö
E__inference_dense_92_layer_call_and_return_conditional_losses_1321280

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
à	
Å
6__inference_Flatten4Layers_Mixed_layer_call_fn_1321166

inputs
unknown://
	unknown_0:/
	unknown_1:/ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1320913o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ/: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs
¦	
¶
%__inference_signature_wrapper_1321145
input_34
unknown://
	unknown_0:/
	unknown_1:/ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_34unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_1320832o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ/: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
"
_user_specified_name
input_34
ö1
ë
"__inference__wrapped_model_1320832
input_34N
<flatten4layers_mixed_dense_92_matmul_readvariableop_resource://K
=flatten4layers_mixed_dense_92_biasadd_readvariableop_resource:/N
<flatten4layers_mixed_dense_93_matmul_readvariableop_resource:/ K
=flatten4layers_mixed_dense_93_biasadd_readvariableop_resource: N
<flatten4layers_mixed_dense_94_matmul_readvariableop_resource:  K
=flatten4layers_mixed_dense_94_biasadd_readvariableop_resource: N
<flatten4layers_mixed_dense_95_matmul_readvariableop_resource: K
=flatten4layers_mixed_dense_95_biasadd_readvariableop_resource:
identity¢4Flatten4Layers_Mixed/dense_92/BiasAdd/ReadVariableOp¢3Flatten4Layers_Mixed/dense_92/MatMul/ReadVariableOp¢4Flatten4Layers_Mixed/dense_93/BiasAdd/ReadVariableOp¢3Flatten4Layers_Mixed/dense_93/MatMul/ReadVariableOp¢4Flatten4Layers_Mixed/dense_94/BiasAdd/ReadVariableOp¢3Flatten4Layers_Mixed/dense_94/MatMul/ReadVariableOp¢4Flatten4Layers_Mixed/dense_95/BiasAdd/ReadVariableOp¢3Flatten4Layers_Mixed/dense_95/MatMul/ReadVariableOpv
%Flatten4Layers_Mixed/flatten_25/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ/   
'Flatten4Layers_Mixed/flatten_25/ReshapeReshapeinput_34.Flatten4Layers_Mixed/flatten_25/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/°
3Flatten4Layers_Mixed/dense_92/MatMul/ReadVariableOpReadVariableOp<flatten4layers_mixed_dense_92_matmul_readvariableop_resource*
_output_shapes

://*
dtype0Ï
$Flatten4Layers_Mixed/dense_92/MatMulMatMul0Flatten4Layers_Mixed/flatten_25/Reshape:output:0;Flatten4Layers_Mixed/dense_92/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/®
4Flatten4Layers_Mixed/dense_92/BiasAdd/ReadVariableOpReadVariableOp=flatten4layers_mixed_dense_92_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0Ð
%Flatten4Layers_Mixed/dense_92/BiasAddBiasAdd.Flatten4Layers_Mixed/dense_92/MatMul:product:0<Flatten4Layers_Mixed/dense_92/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
"Flatten4Layers_Mixed/dense_92/ReluRelu.Flatten4Layers_Mixed/dense_92/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/°
3Flatten4Layers_Mixed/dense_93/MatMul/ReadVariableOpReadVariableOp<flatten4layers_mixed_dense_93_matmul_readvariableop_resource*
_output_shapes

:/ *
dtype0Ï
$Flatten4Layers_Mixed/dense_93/MatMulMatMul0Flatten4Layers_Mixed/dense_92/Relu:activations:0;Flatten4Layers_Mixed/dense_93/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ®
4Flatten4Layers_Mixed/dense_93/BiasAdd/ReadVariableOpReadVariableOp=flatten4layers_mixed_dense_93_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ð
%Flatten4Layers_Mixed/dense_93/BiasAddBiasAdd.Flatten4Layers_Mixed/dense_93/MatMul:product:0<Flatten4Layers_Mixed/dense_93/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
3Flatten4Layers_Mixed/dense_94/MatMul/ReadVariableOpReadVariableOp<flatten4layers_mixed_dense_94_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Í
$Flatten4Layers_Mixed/dense_94/MatMulMatMul.Flatten4Layers_Mixed/dense_93/BiasAdd:output:0;Flatten4Layers_Mixed/dense_94/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ®
4Flatten4Layers_Mixed/dense_94/BiasAdd/ReadVariableOpReadVariableOp=flatten4layers_mixed_dense_94_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ð
%Flatten4Layers_Mixed/dense_94/BiasAddBiasAdd.Flatten4Layers_Mixed/dense_94/MatMul:product:0<Flatten4Layers_Mixed/dense_94/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
3Flatten4Layers_Mixed/dense_95/MatMul/ReadVariableOpReadVariableOp<flatten4layers_mixed_dense_95_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Í
$Flatten4Layers_Mixed/dense_95/MatMulMatMul.Flatten4Layers_Mixed/dense_94/BiasAdd:output:0;Flatten4Layers_Mixed/dense_95/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4Flatten4Layers_Mixed/dense_95/BiasAdd/ReadVariableOpReadVariableOp=flatten4layers_mixed_dense_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%Flatten4Layers_Mixed/dense_95/BiasAddBiasAdd.Flatten4Layers_Mixed/dense_95/MatMul:product:0<Flatten4Layers_Mixed/dense_95/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
IdentityIdentity.Flatten4Layers_Mixed/dense_95/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
NoOpNoOp5^Flatten4Layers_Mixed/dense_92/BiasAdd/ReadVariableOp4^Flatten4Layers_Mixed/dense_92/MatMul/ReadVariableOp5^Flatten4Layers_Mixed/dense_93/BiasAdd/ReadVariableOp4^Flatten4Layers_Mixed/dense_93/MatMul/ReadVariableOp5^Flatten4Layers_Mixed/dense_94/BiasAdd/ReadVariableOp4^Flatten4Layers_Mixed/dense_94/MatMul/ReadVariableOp5^Flatten4Layers_Mixed/dense_95/BiasAdd/ReadVariableOp4^Flatten4Layers_Mixed/dense_95/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ/: : : : : : : : 2l
4Flatten4Layers_Mixed/dense_92/BiasAdd/ReadVariableOp4Flatten4Layers_Mixed/dense_92/BiasAdd/ReadVariableOp2j
3Flatten4Layers_Mixed/dense_92/MatMul/ReadVariableOp3Flatten4Layers_Mixed/dense_92/MatMul/ReadVariableOp2l
4Flatten4Layers_Mixed/dense_93/BiasAdd/ReadVariableOp4Flatten4Layers_Mixed/dense_93/BiasAdd/ReadVariableOp2j
3Flatten4Layers_Mixed/dense_93/MatMul/ReadVariableOp3Flatten4Layers_Mixed/dense_93/MatMul/ReadVariableOp2l
4Flatten4Layers_Mixed/dense_94/BiasAdd/ReadVariableOp4Flatten4Layers_Mixed/dense_94/BiasAdd/ReadVariableOp2j
3Flatten4Layers_Mixed/dense_94/MatMul/ReadVariableOp3Flatten4Layers_Mixed/dense_94/MatMul/ReadVariableOp2l
4Flatten4Layers_Mixed/dense_95/BiasAdd/ReadVariableOp4Flatten4Layers_Mixed/dense_95/BiasAdd/ReadVariableOp2j
3Flatten4Layers_Mixed/dense_95/MatMul/ReadVariableOp3Flatten4Layers_Mixed/dense_95/MatMul/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
"
_user_specified_name
input_34
¹C
ñ
 __inference__traced_save_1321453
file_prefix.
*savev2_dense_92_kernel_read_readvariableop,
(savev2_dense_92_bias_read_readvariableop.
*savev2_dense_93_kernel_read_readvariableop,
(savev2_dense_93_bias_read_readvariableop.
*savev2_dense_94_kernel_read_readvariableop,
(savev2_dense_94_bias_read_readvariableop.
*savev2_dense_95_kernel_read_readvariableop,
(savev2_dense_95_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_92_kernel_m_read_readvariableop3
/savev2_adam_dense_92_bias_m_read_readvariableop5
1savev2_adam_dense_93_kernel_m_read_readvariableop3
/savev2_adam_dense_93_bias_m_read_readvariableop5
1savev2_adam_dense_94_kernel_m_read_readvariableop3
/savev2_adam_dense_94_bias_m_read_readvariableop5
1savev2_adam_dense_95_kernel_m_read_readvariableop3
/savev2_adam_dense_95_bias_m_read_readvariableop5
1savev2_adam_dense_92_kernel_v_read_readvariableop3
/savev2_adam_dense_92_bias_v_read_readvariableop5
1savev2_adam_dense_93_kernel_v_read_readvariableop3
/savev2_adam_dense_93_bias_v_read_readvariableop5
1savev2_adam_dense_94_kernel_v_read_readvariableop3
/savev2_adam_dense_94_bias_v_read_readvariableop5
1savev2_adam_dense_95_kernel_v_read_readvariableop3
/savev2_adam_dense_95_bias_v_read_readvariableop
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
: Ã
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*ì
valueâBß B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH­
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ×
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_92_kernel_read_readvariableop(savev2_dense_92_bias_read_readvariableop*savev2_dense_93_kernel_read_readvariableop(savev2_dense_93_bias_read_readvariableop*savev2_dense_94_kernel_read_readvariableop(savev2_dense_94_bias_read_readvariableop*savev2_dense_95_kernel_read_readvariableop(savev2_dense_95_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_92_kernel_m_read_readvariableop/savev2_adam_dense_92_bias_m_read_readvariableop1savev2_adam_dense_93_kernel_m_read_readvariableop/savev2_adam_dense_93_bias_m_read_readvariableop1savev2_adam_dense_94_kernel_m_read_readvariableop/savev2_adam_dense_94_bias_m_read_readvariableop1savev2_adam_dense_95_kernel_m_read_readvariableop/savev2_adam_dense_95_bias_m_read_readvariableop1savev2_adam_dense_92_kernel_v_read_readvariableop/savev2_adam_dense_92_bias_v_read_readvariableop1savev2_adam_dense_93_kernel_v_read_readvariableop/savev2_adam_dense_93_bias_v_read_readvariableop1savev2_adam_dense_94_kernel_v_read_readvariableop/savev2_adam_dense_94_bias_v_read_readvariableop1savev2_adam_dense_95_kernel_v_read_readvariableop/savev2_adam_dense_95_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	
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

identity_1Identity_1:output:0*ç
_input_shapesÕ
Ò: ://:/:/ : :  : : :: : : : : : : ://:/:/ : :  : : :://:/:/ : :  : : :: 2(
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

:/ : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::	
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
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

://: 

_output_shapes
:/:$ 

_output_shapes

:/ : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

://: 

_output_shapes
:/:$ 

_output_shapes

:/ : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
:: 

_output_shapes
: 
¿
c
G__inference_flatten_25_layer_call_and_return_conditional_losses_1320845

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
à	
Å
6__inference_Flatten4Layers_Mixed_layer_call_fn_1321187

inputs
unknown://
	unknown_0:/
	unknown_1:/ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1321026o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ/: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ/
 
_user_specified_nameinputs


ö
E__inference_dense_92_layer_call_and_return_conditional_losses_1320858

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
 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*±
serving_default
A
input_345
serving_default_input_34:0ÿÿÿÿÿÿÿÿÿ/<
dense_950
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ñ

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
»
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias"
_tf_keras_layer
»
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias"
_tf_keras_layer
X
0
1
#2
$3
+4
,5
36
47"
trackable_list_wrapper
X
0
1
#2
$3
+4
,5
36
47"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object

:trace_0
;trace_1
<trace_2
=trace_32¢
6__inference_Flatten4Layers_Mixed_layer_call_fn_1320932
6__inference_Flatten4Layers_Mixed_layer_call_fn_1321166
6__inference_Flatten4Layers_Mixed_layer_call_fn_1321187
6__inference_Flatten4Layers_Mixed_layer_call_fn_1321066¿
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
 z:trace_0z;trace_1z<trace_2z=trace_3
ù
>trace_0
?trace_1
@trace_2
Atrace_32
Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1321218
Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1321249
Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1321091
Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1321116¿
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
 z>trace_0z?trace_1z@trace_2zAtrace_3
ÎBË
"__inference__wrapped_model_1320832input_34"
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
ã
Biter

Cbeta_1

Dbeta_2
	Edecay
Flearning_ratempmq#mr$ms+mt,mu3mv4mwvxvy#vz$v{+v|,v}3v~4v"
	optimizer
,
Gserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ð
Mtrace_02Ó
,__inference_flatten_25_layer_call_fn_1321254¢
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
 zMtrace_0

Ntrace_02î
G__inference_flatten_25_layer_call_and_return_conditional_losses_1321260¢
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
 zNtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
î
Ttrace_02Ñ
*__inference_dense_92_layer_call_fn_1321269¢
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
 zTtrace_0

Utrace_02ì
E__inference_dense_92_layer_call_and_return_conditional_losses_1321280¢
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
 zUtrace_0
!://2dense_92/kernel
:/2dense_92/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
î
[trace_02Ñ
*__inference_dense_93_layer_call_fn_1321289¢
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
 z[trace_0

\trace_02ì
E__inference_dense_93_layer_call_and_return_conditional_losses_1321299¢
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
 z\trace_0
!:/ 2dense_93/kernel
: 2dense_93/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
î
btrace_02Ñ
*__inference_dense_94_layer_call_fn_1321308¢
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
 zbtrace_0

ctrace_02ì
E__inference_dense_94_layer_call_and_return_conditional_losses_1321318¢
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
 zctrace_0
!:  2dense_94/kernel
: 2dense_94/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
­
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
î
itrace_02Ñ
*__inference_dense_95_layer_call_fn_1321327¢
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
 zitrace_0

jtrace_02ì
E__inference_dense_95_layer_call_and_return_conditional_losses_1321337¢
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
 zjtrace_0
!: 2dense_95/kernel
:2dense_95/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
'
k0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
6__inference_Flatten4Layers_Mixed_layer_call_fn_1320932input_34"¿
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
6__inference_Flatten4Layers_Mixed_layer_call_fn_1321166inputs"¿
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
6__inference_Flatten4Layers_Mixed_layer_call_fn_1321187inputs"¿
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
6__inference_Flatten4Layers_Mixed_layer_call_fn_1321066input_34"¿
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
Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1321218inputs"¿
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
Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1321249inputs"¿
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
Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1321091input_34"¿
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
Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1321116input_34"¿
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
%__inference_signature_wrapper_1321145input_34"
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
,__inference_flatten_25_layer_call_fn_1321254inputs"¢
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
G__inference_flatten_25_layer_call_and_return_conditional_losses_1321260inputs"¢
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
*__inference_dense_92_layer_call_fn_1321269inputs"¢
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
E__inference_dense_92_layer_call_and_return_conditional_losses_1321280inputs"¢
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
*__inference_dense_93_layer_call_fn_1321289inputs"¢
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
E__inference_dense_93_layer_call_and_return_conditional_losses_1321299inputs"¢
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
*__inference_dense_94_layer_call_fn_1321308inputs"¢
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
E__inference_dense_94_layer_call_and_return_conditional_losses_1321318inputs"¢
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
*__inference_dense_95_layer_call_fn_1321327inputs"¢
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
E__inference_dense_95_layer_call_and_return_conditional_losses_1321337inputs"¢
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
l	variables
m	keras_api
	ntotal
	ocount"
_tf_keras_metric
.
n0
o1"
trackable_list_wrapper
-
l	variables"
_generic_user_object
:  (2total
:  (2count
&:$//2Adam/dense_92/kernel/m
 :/2Adam/dense_92/bias/m
&:$/ 2Adam/dense_93/kernel/m
 : 2Adam/dense_93/bias/m
&:$  2Adam/dense_94/kernel/m
 : 2Adam/dense_94/bias/m
&:$ 2Adam/dense_95/kernel/m
 :2Adam/dense_95/bias/m
&:$//2Adam/dense_92/kernel/v
 :/2Adam/dense_92/bias/v
&:$/ 2Adam/dense_93/kernel/v
 : 2Adam/dense_93/bias/v
&:$  2Adam/dense_94/kernel/v
 : 2Adam/dense_94/bias/v
&:$ 2Adam/dense_95/kernel/v
 :2Adam/dense_95/bias/vÅ
Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1321091p#$+,34=¢:
3¢0
&#
input_34ÿÿÿÿÿÿÿÿÿ/
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1321116p#$+,34=¢:
3¢0
&#
input_34ÿÿÿÿÿÿÿÿÿ/
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ã
Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1321218n#$+,34;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ/
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ã
Q__inference_Flatten4Layers_Mixed_layer_call_and_return_conditional_losses_1321249n#$+,34;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ/
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
6__inference_Flatten4Layers_Mixed_layer_call_fn_1320932c#$+,34=¢:
3¢0
&#
input_34ÿÿÿÿÿÿÿÿÿ/
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
6__inference_Flatten4Layers_Mixed_layer_call_fn_1321066c#$+,34=¢:
3¢0
&#
input_34ÿÿÿÿÿÿÿÿÿ/
p

 
ª "ÿÿÿÿÿÿÿÿÿ
6__inference_Flatten4Layers_Mixed_layer_call_fn_1321166a#$+,34;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ/
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
6__inference_Flatten4Layers_Mixed_layer_call_fn_1321187a#$+,34;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ/
p

 
ª "ÿÿÿÿÿÿÿÿÿ
"__inference__wrapped_model_1320832v#$+,345¢2
+¢(
&#
input_34ÿÿÿÿÿÿÿÿÿ/
ª "3ª0
.
dense_95"
dense_95ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_92_layer_call_and_return_conditional_losses_1321280\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ/
 }
*__inference_dense_92_layer_call_fn_1321269O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "ÿÿÿÿÿÿÿÿÿ/¥
E__inference_dense_93_layer_call_and_return_conditional_losses_1321299\#$/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 }
*__inference_dense_93_layer_call_fn_1321289O#$/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ/
ª "ÿÿÿÿÿÿÿÿÿ ¥
E__inference_dense_94_layer_call_and_return_conditional_losses_1321318\+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 }
*__inference_dense_94_layer_call_fn_1321308O+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ¥
E__inference_dense_95_layer_call_and_return_conditional_losses_1321337\34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_95_layer_call_fn_1321327O34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_flatten_25_layer_call_and_return_conditional_losses_1321260\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ/
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ/
 
,__inference_flatten_25_layer_call_fn_1321254O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ/
ª "ÿÿÿÿÿÿÿÿÿ/¬
%__inference_signature_wrapper_1321145#$+,34A¢>
¢ 
7ª4
2
input_34&#
input_34ÿÿÿÿÿÿÿÿÿ/"3ª0
.
dense_95"
dense_95ÿÿÿÿÿÿÿÿÿ