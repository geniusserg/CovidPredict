
Ρ‘
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
Α
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
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8αγ

Adam/dense_86/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_86/bias/v
y
(Adam/dense_86/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_86/bias/v*
_output_shapes
:*
dtype0

Adam/dense_86/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*'
shared_nameAdam/dense_86/kernel/v

*Adam/dense_86/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_86/kernel/v*
_output_shapes

:/*
dtype0

Adam/dense_85/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_85/bias/v
y
(Adam/dense_85/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_85/bias/v*
_output_shapes
:/*
dtype0

Adam/dense_85/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*'
shared_nameAdam/dense_85/kernel/v

*Adam/dense_85/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_85/kernel/v*
_output_shapes

://*
dtype0

Adam/dense_84/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_84/bias/v
y
(Adam/dense_84/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_84/bias/v*
_output_shapes
:/*
dtype0

Adam/dense_84/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:^/*'
shared_nameAdam/dense_84/kernel/v

*Adam/dense_84/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_84/kernel/v*
_output_shapes

:^/*
dtype0

Adam/dense_83/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:^*%
shared_nameAdam/dense_83/bias/v
y
(Adam/dense_83/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_83/bias/v*
_output_shapes
:^*
dtype0

Adam/dense_83/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:^^*'
shared_nameAdam/dense_83/kernel/v

*Adam/dense_83/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_83/kernel/v*
_output_shapes

:^^*
dtype0

Adam/dense_86/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_86/bias/m
y
(Adam/dense_86/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_86/bias/m*
_output_shapes
:*
dtype0

Adam/dense_86/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*'
shared_nameAdam/dense_86/kernel/m

*Adam/dense_86/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_86/kernel/m*
_output_shapes

:/*
dtype0

Adam/dense_85/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_85/bias/m
y
(Adam/dense_85/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_85/bias/m*
_output_shapes
:/*
dtype0

Adam/dense_85/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*'
shared_nameAdam/dense_85/kernel/m

*Adam/dense_85/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_85/kernel/m*
_output_shapes

://*
dtype0

Adam/dense_84/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/dense_84/bias/m
y
(Adam/dense_84/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_84/bias/m*
_output_shapes
:/*
dtype0

Adam/dense_84/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:^/*'
shared_nameAdam/dense_84/kernel/m

*Adam/dense_84/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_84/kernel/m*
_output_shapes

:^/*
dtype0

Adam/dense_83/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:^*%
shared_nameAdam/dense_83/bias/m
y
(Adam/dense_83/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_83/bias/m*
_output_shapes
:^*
dtype0

Adam/dense_83/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:^^*'
shared_nameAdam/dense_83/kernel/m

*Adam/dense_83/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_83/kernel/m*
_output_shapes

:^^*
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
dense_86/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_86/bias
k
!dense_86/bias/Read/ReadVariableOpReadVariableOpdense_86/bias*
_output_shapes
:*
dtype0
z
dense_86/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/* 
shared_namedense_86/kernel
s
#dense_86/kernel/Read/ReadVariableOpReadVariableOpdense_86/kernel*
_output_shapes

:/*
dtype0
r
dense_85/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namedense_85/bias
k
!dense_85/bias/Read/ReadVariableOpReadVariableOpdense_85/bias*
_output_shapes
:/*
dtype0
z
dense_85/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
://* 
shared_namedense_85/kernel
s
#dense_85/kernel/Read/ReadVariableOpReadVariableOpdense_85/kernel*
_output_shapes

://*
dtype0
r
dense_84/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namedense_84/bias
k
!dense_84/bias/Read/ReadVariableOpReadVariableOpdense_84/bias*
_output_shapes
:/*
dtype0
z
dense_84/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:^/* 
shared_namedense_84/kernel
s
#dense_84/kernel/Read/ReadVariableOpReadVariableOpdense_84/kernel*
_output_shapes

:^/*
dtype0
r
dense_83/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:^*
shared_namedense_83/bias
k
!dense_83/bias/Read/ReadVariableOpReadVariableOpdense_83/bias*
_output_shapes
:^*
dtype0
z
dense_83/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:^^* 
shared_namedense_83/kernel
s
#dense_83/kernel/Read/ReadVariableOpReadVariableOpdense_83/kernel*
_output_shapes

:^^*
dtype0

serving_default_input_57Placeholder*+
_output_shapes
:?????????/*
dtype0* 
shape:?????????/
Δ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_57dense_83/kerneldense_83/biasdense_84/kerneldense_84/biasdense_85/kerneldense_85/biasdense_86/kerneldense_86/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1526148

NoOpNoOp
°;
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*λ:
valueα:Bή: BΧ:
υ
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
Τ
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
VARIABLE_VALUEdense_83/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_83/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_84/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_84/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_85/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_85/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_86/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_86/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_83/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_83/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_84/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_84/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_85/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_85/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_86/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_86/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_83/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_83/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_84/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_84/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_85/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_85/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_86/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_86/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
κ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_83/kernel/Read/ReadVariableOp!dense_83/bias/Read/ReadVariableOp#dense_84/kernel/Read/ReadVariableOp!dense_84/bias/Read/ReadVariableOp#dense_85/kernel/Read/ReadVariableOp!dense_85/bias/Read/ReadVariableOp#dense_86/kernel/Read/ReadVariableOp!dense_86/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_83/kernel/m/Read/ReadVariableOp(Adam/dense_83/bias/m/Read/ReadVariableOp*Adam/dense_84/kernel/m/Read/ReadVariableOp(Adam/dense_84/bias/m/Read/ReadVariableOp*Adam/dense_85/kernel/m/Read/ReadVariableOp(Adam/dense_85/bias/m/Read/ReadVariableOp*Adam/dense_86/kernel/m/Read/ReadVariableOp(Adam/dense_86/bias/m/Read/ReadVariableOp*Adam/dense_83/kernel/v/Read/ReadVariableOp(Adam/dense_83/bias/v/Read/ReadVariableOp*Adam/dense_84/kernel/v/Read/ReadVariableOp(Adam/dense_84/bias/v/Read/ReadVariableOp*Adam/dense_85/kernel/v/Read/ReadVariableOp(Adam/dense_85/bias/v/Read/ReadVariableOp*Adam/dense_86/kernel/v/Read/ReadVariableOp(Adam/dense_86/bias/v/Read/ReadVariableOpConst*,
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
 __inference__traced_save_1526462
ω
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_83/kerneldense_83/biasdense_84/kerneldense_84/biasdense_85/kerneldense_85/biasdense_86/kerneldense_86/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_83/kernel/mAdam/dense_83/bias/mAdam/dense_84/kernel/mAdam/dense_84/bias/mAdam/dense_85/kernel/mAdam/dense_85/bias/mAdam/dense_86/kernel/mAdam/dense_86/bias/mAdam/dense_83/kernel/vAdam/dense_83/bias/vAdam/dense_84/kernel/vAdam/dense_84/bias/vAdam/dense_85/kernel/vAdam/dense_85/bias/vAdam/dense_86/kernel/vAdam/dense_86/bias/v*+
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
#__inference__traced_restore_1526565ΗΥ
«
H
,__inference_flatten_11_layer_call_fn_1526261

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
:?????????^* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_1525846`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????^"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs
Δ

*__inference_dense_84_layer_call_fn_1526296

inputs
unknown:^/
	unknown_0:/
identity’StatefulPartitionedCallΪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_84_layer_call_and_return_conditional_losses_1525876o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????/`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
ζ	
Η
6__inference_Flatten3Layers_Mixed_layer_call_fn_1526069
input_57
unknown:^^
	unknown_0:^
	unknown_1:^/
	unknown_2:/
	unknown_3://
	unknown_4:/
	unknown_5:/
	unknown_6:
identity’StatefulPartitionedCallΆ
StatefulPartitionedCallStatefulPartitionedCallinput_57unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1526029o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????/: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????/
"
_user_specified_name
input_57
Ώ
c
G__inference_flatten_11_layer_call_and_return_conditional_losses_1526267

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????^   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????^X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????^"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs
Θ	
φ
E__inference_dense_86_layer_call_and_return_conditional_losses_1526346

inputs0
matmul_readvariableop_resource:/-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
Δ

*__inference_dense_86_layer_call_fn_1526336

inputs
unknown:/
	unknown_0:
identity’StatefulPartitionedCallΪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_86_layer_call_and_return_conditional_losses_1525909o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
ΰ	
Ε
6__inference_Flatten3Layers_Mixed_layer_call_fn_1526169

inputs
unknown:^^
	unknown_0:^
	unknown_1:^/
	unknown_2:/
	unknown_3://
	unknown_4:/
	unknown_5:/
	unknown_6:
identity’StatefulPartitionedCall΄
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1525916o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????/: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs


φ
E__inference_dense_84_layer_call_and_return_conditional_losses_1525876

inputs0
matmul_readvariableop_resource:^/-
biasadd_readvariableop_resource:/
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^/*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????/a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????/w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs


φ
E__inference_dense_85_layer_call_and_return_conditional_losses_1526327

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????/a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????/w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
ξ

Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1526029

inputs"
dense_83_1526008:^^
dense_83_1526010:^"
dense_84_1526013:^/
dense_84_1526015:/"
dense_85_1526018://
dense_85_1526020:/"
dense_86_1526023:/
dense_86_1526025:
identity’ dense_83/StatefulPartitionedCall’ dense_84/StatefulPartitionedCall’ dense_85/StatefulPartitionedCall’ dense_86/StatefulPartitionedCall½
flatten_11/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_1525846
 dense_83/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_83_1526008dense_83_1526010*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_83_layer_call_and_return_conditional_losses_1525859
 dense_84/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0dense_84_1526013dense_84_1526015*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_84_layer_call_and_return_conditional_losses_1525876
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0dense_85_1526018dense_85_1526020*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_85_layer_call_and_return_conditional_losses_1525893
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0dense_86_1526023dense_86_1526025*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_86_layer_call_and_return_conditional_losses_1525909x
IdentityIdentity)dense_86/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_83/StatefulPartitionedCall!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????/: : : : : : : : 2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs
Ϋ%
Θ
Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1526256

inputs9
'dense_83_matmul_readvariableop_resource:^^6
(dense_83_biasadd_readvariableop_resource:^9
'dense_84_matmul_readvariableop_resource:^/6
(dense_84_biasadd_readvariableop_resource:/9
'dense_85_matmul_readvariableop_resource://6
(dense_85_biasadd_readvariableop_resource:/9
'dense_86_matmul_readvariableop_resource:/6
(dense_86_biasadd_readvariableop_resource:
identity’dense_83/BiasAdd/ReadVariableOp’dense_83/MatMul/ReadVariableOp’dense_84/BiasAdd/ReadVariableOp’dense_84/MatMul/ReadVariableOp’dense_85/BiasAdd/ReadVariableOp’dense_85/MatMul/ReadVariableOp’dense_86/BiasAdd/ReadVariableOp’dense_86/MatMul/ReadVariableOpa
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????^   r
flatten_11/ReshapeReshapeinputsflatten_11/Const:output:0*
T0*'
_output_shapes
:?????????^
dense_83/MatMul/ReadVariableOpReadVariableOp'dense_83_matmul_readvariableop_resource*
_output_shapes

:^^*
dtype0
dense_83/MatMulMatMulflatten_11/Reshape:output:0&dense_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
dense_83/BiasAdd/ReadVariableOpReadVariableOp(dense_83_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype0
dense_83/BiasAddBiasAdddense_83/MatMul:product:0'dense_83/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^b
dense_83/ReluReludense_83/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^
dense_84/MatMul/ReadVariableOpReadVariableOp'dense_84_matmul_readvariableop_resource*
_output_shapes

:^/*
dtype0
dense_84/MatMulMatMuldense_83/Relu:activations:0&dense_84/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/
dense_84/BiasAdd/ReadVariableOpReadVariableOp(dense_84_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0
dense_84/BiasAddBiasAdddense_84/MatMul:product:0'dense_84/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/b
dense_84/ReluReludense_84/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/
dense_85/MatMul/ReadVariableOpReadVariableOp'dense_85_matmul_readvariableop_resource*
_output_shapes

://*
dtype0
dense_85/MatMulMatMuldense_84/Relu:activations:0&dense_85/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/
dense_85/BiasAdd/ReadVariableOpReadVariableOp(dense_85_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0
dense_85/BiasAddBiasAdddense_85/MatMul:product:0'dense_85/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/b
dense_85/ReluReludense_85/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource*
_output_shapes

:/*
dtype0
dense_86/MatMulMatMuldense_85/Relu:activations:0&dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_86/BiasAdd/ReadVariableOpReadVariableOp(dense_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_86/BiasAddBiasAdddense_86/MatMul:product:0'dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_86/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_83/BiasAdd/ReadVariableOp^dense_83/MatMul/ReadVariableOp ^dense_84/BiasAdd/ReadVariableOp^dense_84/MatMul/ReadVariableOp ^dense_85/BiasAdd/ReadVariableOp^dense_85/MatMul/ReadVariableOp ^dense_86/BiasAdd/ReadVariableOp^dense_86/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????/: : : : : : : : 2B
dense_83/BiasAdd/ReadVariableOpdense_83/BiasAdd/ReadVariableOp2@
dense_83/MatMul/ReadVariableOpdense_83/MatMul/ReadVariableOp2B
dense_84/BiasAdd/ReadVariableOpdense_84/BiasAdd/ReadVariableOp2@
dense_84/MatMul/ReadVariableOpdense_84/MatMul/ReadVariableOp2B
dense_85/BiasAdd/ReadVariableOpdense_85/BiasAdd/ReadVariableOp2@
dense_85/MatMul/ReadVariableOpdense_85/MatMul/ReadVariableOp2B
dense_86/BiasAdd/ReadVariableOpdense_86/BiasAdd/ReadVariableOp2@
dense_86/MatMul/ReadVariableOpdense_86/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs
τ

Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1526119
input_57"
dense_83_1526098:^^
dense_83_1526100:^"
dense_84_1526103:^/
dense_84_1526105:/"
dense_85_1526108://
dense_85_1526110:/"
dense_86_1526113:/
dense_86_1526115:
identity’ dense_83/StatefulPartitionedCall’ dense_84/StatefulPartitionedCall’ dense_85/StatefulPartitionedCall’ dense_86/StatefulPartitionedCallΏ
flatten_11/PartitionedCallPartitionedCallinput_57*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_1525846
 dense_83/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_83_1526098dense_83_1526100*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_83_layer_call_and_return_conditional_losses_1525859
 dense_84/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0dense_84_1526103dense_84_1526105*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_84_layer_call_and_return_conditional_losses_1525876
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0dense_85_1526108dense_85_1526110*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_85_layer_call_and_return_conditional_losses_1525893
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0dense_86_1526113dense_86_1526115*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_86_layer_call_and_return_conditional_losses_1525909x
IdentityIdentity)dense_86/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_83/StatefulPartitionedCall!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????/: : : : : : : : 2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????/
"
_user_specified_name
input_57
τ

Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1526094
input_57"
dense_83_1526073:^^
dense_83_1526075:^"
dense_84_1526078:^/
dense_84_1526080:/"
dense_85_1526083://
dense_85_1526085:/"
dense_86_1526088:/
dense_86_1526090:
identity’ dense_83/StatefulPartitionedCall’ dense_84/StatefulPartitionedCall’ dense_85/StatefulPartitionedCall’ dense_86/StatefulPartitionedCallΏ
flatten_11/PartitionedCallPartitionedCallinput_57*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_1525846
 dense_83/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_83_1526073dense_83_1526075*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_83_layer_call_and_return_conditional_losses_1525859
 dense_84/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0dense_84_1526078dense_84_1526080*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_84_layer_call_and_return_conditional_losses_1525876
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0dense_85_1526083dense_85_1526085*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_85_layer_call_and_return_conditional_losses_1525893
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0dense_86_1526088dense_86_1526090*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_86_layer_call_and_return_conditional_losses_1525909x
IdentityIdentity)dense_86/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_83/StatefulPartitionedCall!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????/: : : : : : : : 2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????/
"
_user_specified_name
input_57
Ϋ%
Θ
Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1526223

inputs9
'dense_83_matmul_readvariableop_resource:^^6
(dense_83_biasadd_readvariableop_resource:^9
'dense_84_matmul_readvariableop_resource:^/6
(dense_84_biasadd_readvariableop_resource:/9
'dense_85_matmul_readvariableop_resource://6
(dense_85_biasadd_readvariableop_resource:/9
'dense_86_matmul_readvariableop_resource:/6
(dense_86_biasadd_readvariableop_resource:
identity’dense_83/BiasAdd/ReadVariableOp’dense_83/MatMul/ReadVariableOp’dense_84/BiasAdd/ReadVariableOp’dense_84/MatMul/ReadVariableOp’dense_85/BiasAdd/ReadVariableOp’dense_85/MatMul/ReadVariableOp’dense_86/BiasAdd/ReadVariableOp’dense_86/MatMul/ReadVariableOpa
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????^   r
flatten_11/ReshapeReshapeinputsflatten_11/Const:output:0*
T0*'
_output_shapes
:?????????^
dense_83/MatMul/ReadVariableOpReadVariableOp'dense_83_matmul_readvariableop_resource*
_output_shapes

:^^*
dtype0
dense_83/MatMulMatMulflatten_11/Reshape:output:0&dense_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
dense_83/BiasAdd/ReadVariableOpReadVariableOp(dense_83_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype0
dense_83/BiasAddBiasAdddense_83/MatMul:product:0'dense_83/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^b
dense_83/ReluReludense_83/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^
dense_84/MatMul/ReadVariableOpReadVariableOp'dense_84_matmul_readvariableop_resource*
_output_shapes

:^/*
dtype0
dense_84/MatMulMatMuldense_83/Relu:activations:0&dense_84/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/
dense_84/BiasAdd/ReadVariableOpReadVariableOp(dense_84_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0
dense_84/BiasAddBiasAdddense_84/MatMul:product:0'dense_84/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/b
dense_84/ReluReludense_84/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/
dense_85/MatMul/ReadVariableOpReadVariableOp'dense_85_matmul_readvariableop_resource*
_output_shapes

://*
dtype0
dense_85/MatMulMatMuldense_84/Relu:activations:0&dense_85/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/
dense_85/BiasAdd/ReadVariableOpReadVariableOp(dense_85_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0
dense_85/BiasAddBiasAdddense_85/MatMul:product:0'dense_85/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/b
dense_85/ReluReludense_85/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource*
_output_shapes

:/*
dtype0
dense_86/MatMulMatMuldense_85/Relu:activations:0&dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_86/BiasAdd/ReadVariableOpReadVariableOp(dense_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_86/BiasAddBiasAdddense_86/MatMul:product:0'dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_86/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_83/BiasAdd/ReadVariableOp^dense_83/MatMul/ReadVariableOp ^dense_84/BiasAdd/ReadVariableOp^dense_84/MatMul/ReadVariableOp ^dense_85/BiasAdd/ReadVariableOp^dense_85/MatMul/ReadVariableOp ^dense_86/BiasAdd/ReadVariableOp^dense_86/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????/: : : : : : : : 2B
dense_83/BiasAdd/ReadVariableOpdense_83/BiasAdd/ReadVariableOp2@
dense_83/MatMul/ReadVariableOpdense_83/MatMul/ReadVariableOp2B
dense_84/BiasAdd/ReadVariableOpdense_84/BiasAdd/ReadVariableOp2@
dense_84/MatMul/ReadVariableOpdense_84/MatMul/ReadVariableOp2B
dense_85/BiasAdd/ReadVariableOpdense_85/BiasAdd/ReadVariableOp2@
dense_85/MatMul/ReadVariableOpdense_85/MatMul/ReadVariableOp2B
dense_86/BiasAdd/ReadVariableOpdense_86/BiasAdd/ReadVariableOp2@
dense_86/MatMul/ReadVariableOpdense_86/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs
}
Ρ
#__inference__traced_restore_1526565
file_prefix2
 assignvariableop_dense_83_kernel:^^.
 assignvariableop_1_dense_83_bias:^4
"assignvariableop_2_dense_84_kernel:^/.
 assignvariableop_3_dense_84_bias:/4
"assignvariableop_4_dense_85_kernel://.
 assignvariableop_5_dense_85_bias:/4
"assignvariableop_6_dense_86_kernel:/.
 assignvariableop_7_dense_86_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: <
*assignvariableop_15_adam_dense_83_kernel_m:^^6
(assignvariableop_16_adam_dense_83_bias_m:^<
*assignvariableop_17_adam_dense_84_kernel_m:^/6
(assignvariableop_18_adam_dense_84_bias_m:/<
*assignvariableop_19_adam_dense_85_kernel_m://6
(assignvariableop_20_adam_dense_85_bias_m:/<
*assignvariableop_21_adam_dense_86_kernel_m:/6
(assignvariableop_22_adam_dense_86_bias_m:<
*assignvariableop_23_adam_dense_83_kernel_v:^^6
(assignvariableop_24_adam_dense_83_bias_v:^<
*assignvariableop_25_adam_dense_84_kernel_v:^/6
(assignvariableop_26_adam_dense_84_bias_v:/<
*assignvariableop_27_adam_dense_85_kernel_v://6
(assignvariableop_28_adam_dense_85_bias_v:/<
*assignvariableop_29_adam_dense_86_kernel_v:/6
(assignvariableop_30_adam_dense_86_bias_v:
identity_32’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9Ζ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*μ
valueβBί B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH°
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Α
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_dense_83_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_83_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_84_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_84_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_85_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_85_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_86_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_86_biasIdentity_7:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_83_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_83_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_84_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_84_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_85_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_85_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_86_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_86_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_83_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_83_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_84_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_84_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_85_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_85_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_86_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_86_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ω
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: ζ
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
4
λ
"__inference__wrapped_model_1525833
input_57N
<flatten3layers_mixed_dense_83_matmul_readvariableop_resource:^^K
=flatten3layers_mixed_dense_83_biasadd_readvariableop_resource:^N
<flatten3layers_mixed_dense_84_matmul_readvariableop_resource:^/K
=flatten3layers_mixed_dense_84_biasadd_readvariableop_resource:/N
<flatten3layers_mixed_dense_85_matmul_readvariableop_resource://K
=flatten3layers_mixed_dense_85_biasadd_readvariableop_resource:/N
<flatten3layers_mixed_dense_86_matmul_readvariableop_resource:/K
=flatten3layers_mixed_dense_86_biasadd_readvariableop_resource:
identity’4Flatten3Layers_Mixed/dense_83/BiasAdd/ReadVariableOp’3Flatten3Layers_Mixed/dense_83/MatMul/ReadVariableOp’4Flatten3Layers_Mixed/dense_84/BiasAdd/ReadVariableOp’3Flatten3Layers_Mixed/dense_84/MatMul/ReadVariableOp’4Flatten3Layers_Mixed/dense_85/BiasAdd/ReadVariableOp’3Flatten3Layers_Mixed/dense_85/MatMul/ReadVariableOp’4Flatten3Layers_Mixed/dense_86/BiasAdd/ReadVariableOp’3Flatten3Layers_Mixed/dense_86/MatMul/ReadVariableOpv
%Flatten3Layers_Mixed/flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????^   
'Flatten3Layers_Mixed/flatten_11/ReshapeReshapeinput_57.Flatten3Layers_Mixed/flatten_11/Const:output:0*
T0*'
_output_shapes
:?????????^°
3Flatten3Layers_Mixed/dense_83/MatMul/ReadVariableOpReadVariableOp<flatten3layers_mixed_dense_83_matmul_readvariableop_resource*
_output_shapes

:^^*
dtype0Ο
$Flatten3Layers_Mixed/dense_83/MatMulMatMul0Flatten3Layers_Mixed/flatten_11/Reshape:output:0;Flatten3Layers_Mixed/dense_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^?
4Flatten3Layers_Mixed/dense_83/BiasAdd/ReadVariableOpReadVariableOp=flatten3layers_mixed_dense_83_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype0Π
%Flatten3Layers_Mixed/dense_83/BiasAddBiasAdd.Flatten3Layers_Mixed/dense_83/MatMul:product:0<Flatten3Layers_Mixed/dense_83/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
"Flatten3Layers_Mixed/dense_83/ReluRelu.Flatten3Layers_Mixed/dense_83/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^°
3Flatten3Layers_Mixed/dense_84/MatMul/ReadVariableOpReadVariableOp<flatten3layers_mixed_dense_84_matmul_readvariableop_resource*
_output_shapes

:^/*
dtype0Ο
$Flatten3Layers_Mixed/dense_84/MatMulMatMul0Flatten3Layers_Mixed/dense_83/Relu:activations:0;Flatten3Layers_Mixed/dense_84/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/?
4Flatten3Layers_Mixed/dense_84/BiasAdd/ReadVariableOpReadVariableOp=flatten3layers_mixed_dense_84_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0Π
%Flatten3Layers_Mixed/dense_84/BiasAddBiasAdd.Flatten3Layers_Mixed/dense_84/MatMul:product:0<Flatten3Layers_Mixed/dense_84/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/
"Flatten3Layers_Mixed/dense_84/ReluRelu.Flatten3Layers_Mixed/dense_84/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/°
3Flatten3Layers_Mixed/dense_85/MatMul/ReadVariableOpReadVariableOp<flatten3layers_mixed_dense_85_matmul_readvariableop_resource*
_output_shapes

://*
dtype0Ο
$Flatten3Layers_Mixed/dense_85/MatMulMatMul0Flatten3Layers_Mixed/dense_84/Relu:activations:0;Flatten3Layers_Mixed/dense_85/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/?
4Flatten3Layers_Mixed/dense_85/BiasAdd/ReadVariableOpReadVariableOp=flatten3layers_mixed_dense_85_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0Π
%Flatten3Layers_Mixed/dense_85/BiasAddBiasAdd.Flatten3Layers_Mixed/dense_85/MatMul:product:0<Flatten3Layers_Mixed/dense_85/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/
"Flatten3Layers_Mixed/dense_85/ReluRelu.Flatten3Layers_Mixed/dense_85/BiasAdd:output:0*
T0*'
_output_shapes
:?????????/°
3Flatten3Layers_Mixed/dense_86/MatMul/ReadVariableOpReadVariableOp<flatten3layers_mixed_dense_86_matmul_readvariableop_resource*
_output_shapes

:/*
dtype0Ο
$Flatten3Layers_Mixed/dense_86/MatMulMatMul0Flatten3Layers_Mixed/dense_85/Relu:activations:0;Flatten3Layers_Mixed/dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
4Flatten3Layers_Mixed/dense_86/BiasAdd/ReadVariableOpReadVariableOp=flatten3layers_mixed_dense_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Π
%Flatten3Layers_Mixed/dense_86/BiasAddBiasAdd.Flatten3Layers_Mixed/dense_86/MatMul:product:0<Flatten3Layers_Mixed/dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????}
IdentityIdentity.Flatten3Layers_Mixed/dense_86/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????ϊ
NoOpNoOp5^Flatten3Layers_Mixed/dense_83/BiasAdd/ReadVariableOp4^Flatten3Layers_Mixed/dense_83/MatMul/ReadVariableOp5^Flatten3Layers_Mixed/dense_84/BiasAdd/ReadVariableOp4^Flatten3Layers_Mixed/dense_84/MatMul/ReadVariableOp5^Flatten3Layers_Mixed/dense_85/BiasAdd/ReadVariableOp4^Flatten3Layers_Mixed/dense_85/MatMul/ReadVariableOp5^Flatten3Layers_Mixed/dense_86/BiasAdd/ReadVariableOp4^Flatten3Layers_Mixed/dense_86/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????/: : : : : : : : 2l
4Flatten3Layers_Mixed/dense_83/BiasAdd/ReadVariableOp4Flatten3Layers_Mixed/dense_83/BiasAdd/ReadVariableOp2j
3Flatten3Layers_Mixed/dense_83/MatMul/ReadVariableOp3Flatten3Layers_Mixed/dense_83/MatMul/ReadVariableOp2l
4Flatten3Layers_Mixed/dense_84/BiasAdd/ReadVariableOp4Flatten3Layers_Mixed/dense_84/BiasAdd/ReadVariableOp2j
3Flatten3Layers_Mixed/dense_84/MatMul/ReadVariableOp3Flatten3Layers_Mixed/dense_84/MatMul/ReadVariableOp2l
4Flatten3Layers_Mixed/dense_85/BiasAdd/ReadVariableOp4Flatten3Layers_Mixed/dense_85/BiasAdd/ReadVariableOp2j
3Flatten3Layers_Mixed/dense_85/MatMul/ReadVariableOp3Flatten3Layers_Mixed/dense_85/MatMul/ReadVariableOp2l
4Flatten3Layers_Mixed/dense_86/BiasAdd/ReadVariableOp4Flatten3Layers_Mixed/dense_86/BiasAdd/ReadVariableOp2j
3Flatten3Layers_Mixed/dense_86/MatMul/ReadVariableOp3Flatten3Layers_Mixed/dense_86/MatMul/ReadVariableOp:U Q
+
_output_shapes
:?????????/
"
_user_specified_name
input_57


φ
E__inference_dense_83_layer_call_and_return_conditional_losses_1526287

inputs0
matmul_readvariableop_resource:^^-
biasadd_readvariableop_resource:^
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^^*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:^*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????^a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????^w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs


φ
E__inference_dense_85_layer_call_and_return_conditional_losses_1525893

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????/a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????/w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
Δ

*__inference_dense_83_layer_call_fn_1526276

inputs
unknown:^^
	unknown_0:^
identity’StatefulPartitionedCallΪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_83_layer_call_and_return_conditional_losses_1525859o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????^`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
Ώ
c
G__inference_flatten_11_layer_call_and_return_conditional_losses_1525846

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????^   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????^X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????^"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs


φ
E__inference_dense_83_layer_call_and_return_conditional_losses_1525859

inputs0
matmul_readvariableop_resource:^^-
biasadd_readvariableop_resource:^
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^^*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:^*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????^a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????^w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
Δ

*__inference_dense_85_layer_call_fn_1526316

inputs
unknown://
	unknown_0:/
identity’StatefulPartitionedCallΪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_85_layer_call_and_return_conditional_losses_1525893o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????/`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
ΉC
ρ
 __inference__traced_save_1526462
file_prefix.
*savev2_dense_83_kernel_read_readvariableop,
(savev2_dense_83_bias_read_readvariableop.
*savev2_dense_84_kernel_read_readvariableop,
(savev2_dense_84_bias_read_readvariableop.
*savev2_dense_85_kernel_read_readvariableop,
(savev2_dense_85_bias_read_readvariableop.
*savev2_dense_86_kernel_read_readvariableop,
(savev2_dense_86_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_83_kernel_m_read_readvariableop3
/savev2_adam_dense_83_bias_m_read_readvariableop5
1savev2_adam_dense_84_kernel_m_read_readvariableop3
/savev2_adam_dense_84_bias_m_read_readvariableop5
1savev2_adam_dense_85_kernel_m_read_readvariableop3
/savev2_adam_dense_85_bias_m_read_readvariableop5
1savev2_adam_dense_86_kernel_m_read_readvariableop3
/savev2_adam_dense_86_bias_m_read_readvariableop5
1savev2_adam_dense_83_kernel_v_read_readvariableop3
/savev2_adam_dense_83_bias_v_read_readvariableop5
1savev2_adam_dense_84_kernel_v_read_readvariableop3
/savev2_adam_dense_84_bias_v_read_readvariableop5
1savev2_adam_dense_85_kernel_v_read_readvariableop3
/savev2_adam_dense_85_bias_v_read_readvariableop5
1savev2_adam_dense_86_kernel_v_read_readvariableop3
/savev2_adam_dense_86_bias_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
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
: Γ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*μ
valueβBί B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH­
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Χ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_83_kernel_read_readvariableop(savev2_dense_83_bias_read_readvariableop*savev2_dense_84_kernel_read_readvariableop(savev2_dense_84_bias_read_readvariableop*savev2_dense_85_kernel_read_readvariableop(savev2_dense_85_bias_read_readvariableop*savev2_dense_86_kernel_read_readvariableop(savev2_dense_86_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_83_kernel_m_read_readvariableop/savev2_adam_dense_83_bias_m_read_readvariableop1savev2_adam_dense_84_kernel_m_read_readvariableop/savev2_adam_dense_84_bias_m_read_readvariableop1savev2_adam_dense_85_kernel_m_read_readvariableop/savev2_adam_dense_85_bias_m_read_readvariableop1savev2_adam_dense_86_kernel_m_read_readvariableop/savev2_adam_dense_86_bias_m_read_readvariableop1savev2_adam_dense_83_kernel_v_read_readvariableop/savev2_adam_dense_83_bias_v_read_readvariableop1savev2_adam_dense_84_kernel_v_read_readvariableop/savev2_adam_dense_84_bias_v_read_readvariableop1savev2_adam_dense_85_kernel_v_read_readvariableop/savev2_adam_dense_85_bias_v_read_readvariableop1savev2_adam_dense_86_kernel_v_read_readvariableop/savev2_adam_dense_86_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*η
_input_shapesΥ
?: :^^:^:^/:/://:/:/:: : : : : : : :^^:^:^/:/://:/:/::^^:^:^/:/://:/:/:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:^^: 

_output_shapes
:^:$ 

_output_shapes

:^/: 

_output_shapes
:/:$ 

_output_shapes

://: 

_output_shapes
:/:$ 

_output_shapes

:/: 
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

:^^: 

_output_shapes
:^:$ 

_output_shapes

:^/: 

_output_shapes
:/:$ 

_output_shapes

://: 

_output_shapes
:/:$ 

_output_shapes

:/: 

_output_shapes
::$ 

_output_shapes

:^^: 

_output_shapes
:^:$ 

_output_shapes

:^/: 

_output_shapes
:/:$ 

_output_shapes

://: 

_output_shapes
:/:$ 

_output_shapes

:/: 

_output_shapes
:: 

_output_shapes
: 
ζ	
Η
6__inference_Flatten3Layers_Mixed_layer_call_fn_1525935
input_57
unknown:^^
	unknown_0:^
	unknown_1:^/
	unknown_2:/
	unknown_3://
	unknown_4:/
	unknown_5:/
	unknown_6:
identity’StatefulPartitionedCallΆ
StatefulPartitionedCallStatefulPartitionedCallinput_57unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1525916o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????/: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????/
"
_user_specified_name
input_57
¦	
Ά
%__inference_signature_wrapper_1526148
input_57
unknown:^^
	unknown_0:^
	unknown_1:^/
	unknown_2:/
	unknown_3://
	unknown_4:/
	unknown_5:/
	unknown_6:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_57unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_1525833o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????/: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????/
"
_user_specified_name
input_57


φ
E__inference_dense_84_layer_call_and_return_conditional_losses_1526307

inputs0
matmul_readvariableop_resource:^/-
biasadd_readvariableop_resource:/
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^/*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????/P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????/a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????/w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
ΰ	
Ε
6__inference_Flatten3Layers_Mixed_layer_call_fn_1526190

inputs
unknown:^^
	unknown_0:^
	unknown_1:^/
	unknown_2:/
	unknown_3://
	unknown_4:/
	unknown_5:/
	unknown_6:
identity’StatefulPartitionedCall΄
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1526029o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????/: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs
Θ	
φ
E__inference_dense_86_layer_call_and_return_conditional_losses_1525909

inputs0
matmul_readvariableop_resource:/-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????/
 
_user_specified_nameinputs
ξ

Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1525916

inputs"
dense_83_1525860:^^
dense_83_1525862:^"
dense_84_1525877:^/
dense_84_1525879:/"
dense_85_1525894://
dense_85_1525896:/"
dense_86_1525910:/
dense_86_1525912:
identity’ dense_83/StatefulPartitionedCall’ dense_84/StatefulPartitionedCall’ dense_85/StatefulPartitionedCall’ dense_86/StatefulPartitionedCall½
flatten_11/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_1525846
 dense_83/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_83_1525860dense_83_1525862*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_83_layer_call_and_return_conditional_losses_1525859
 dense_84/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0dense_84_1525877dense_84_1525879*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_84_layer_call_and_return_conditional_losses_1525876
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0dense_85_1525894dense_85_1525896*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_85_layer_call_and_return_conditional_losses_1525893
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0dense_86_1525910dense_86_1525912*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_86_layer_call_and_return_conditional_losses_1525909x
IdentityIdentity)dense_86/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_83/StatefulPartitionedCall!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????/: : : : : : : : 2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs"΅	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*±
serving_default
A
input_575
serving_default_input_57:0?????????/<
dense_860
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ρ
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
₯
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
Κ
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
=trace_32’
6__inference_Flatten3Layers_Mixed_layer_call_fn_1525935
6__inference_Flatten3Layers_Mixed_layer_call_fn_1526169
6__inference_Flatten3Layers_Mixed_layer_call_fn_1526190
6__inference_Flatten3Layers_Mixed_layer_call_fn_1526069Ώ
Ά²²
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
annotationsͺ *
 z:trace_0z;trace_1z<trace_2z=trace_3
ω
>trace_0
?trace_1
@trace_2
Atrace_32
Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1526223
Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1526256
Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1526094
Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1526119Ώ
Ά²²
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
annotationsͺ *
 z>trace_0z?trace_1z@trace_2zAtrace_3
ΞBΛ
"__inference__wrapped_model_1525833input_57"
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
annotationsͺ *
 
γ
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
π
Mtrace_02Σ
,__inference_flatten_11_layer_call_fn_1526261’
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
annotationsͺ *
 zMtrace_0

Ntrace_02ξ
G__inference_flatten_11_layer_call_and_return_conditional_losses_1526267’
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
annotationsͺ *
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
ξ
Ttrace_02Ρ
*__inference_dense_83_layer_call_fn_1526276’
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
annotationsͺ *
 zTtrace_0

Utrace_02μ
E__inference_dense_83_layer_call_and_return_conditional_losses_1526287’
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
annotationsͺ *
 zUtrace_0
!:^^2dense_83/kernel
:^2dense_83/bias
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
ξ
[trace_02Ρ
*__inference_dense_84_layer_call_fn_1526296’
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
annotationsͺ *
 z[trace_0

\trace_02μ
E__inference_dense_84_layer_call_and_return_conditional_losses_1526307’
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
annotationsͺ *
 z\trace_0
!:^/2dense_84/kernel
:/2dense_84/bias
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
ξ
btrace_02Ρ
*__inference_dense_85_layer_call_fn_1526316’
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
annotationsͺ *
 zbtrace_0

ctrace_02μ
E__inference_dense_85_layer_call_and_return_conditional_losses_1526327’
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
annotationsͺ *
 zctrace_0
!://2dense_85/kernel
:/2dense_85/bias
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
ξ
itrace_02Ρ
*__inference_dense_86_layer_call_fn_1526336’
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
annotationsͺ *
 zitrace_0

jtrace_02μ
E__inference_dense_86_layer_call_and_return_conditional_losses_1526346’
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
annotationsͺ *
 zjtrace_0
!:/2dense_86/kernel
:2dense_86/bias
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
6__inference_Flatten3Layers_Mixed_layer_call_fn_1525935input_57"Ώ
Ά²²
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
annotationsͺ *
 
B
6__inference_Flatten3Layers_Mixed_layer_call_fn_1526169inputs"Ώ
Ά²²
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
annotationsͺ *
 
B
6__inference_Flatten3Layers_Mixed_layer_call_fn_1526190inputs"Ώ
Ά²²
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
annotationsͺ *
 
B
6__inference_Flatten3Layers_Mixed_layer_call_fn_1526069input_57"Ώ
Ά²²
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
annotationsͺ *
 
’B
Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1526223inputs"Ώ
Ά²²
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
annotationsͺ *
 
’B
Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1526256inputs"Ώ
Ά²²
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
annotationsͺ *
 
€B‘
Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1526094input_57"Ώ
Ά²²
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
annotationsͺ *
 
€B‘
Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1526119input_57"Ώ
Ά²²
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
annotationsͺ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ΝBΚ
%__inference_signature_wrapper_1526148input_57"
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
annotationsͺ *
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
ΰBέ
,__inference_flatten_11_layer_call_fn_1526261inputs"’
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
annotationsͺ *
 
ϋBψ
G__inference_flatten_11_layer_call_and_return_conditional_losses_1526267inputs"’
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
annotationsͺ *
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
ήBΫ
*__inference_dense_83_layer_call_fn_1526276inputs"’
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
annotationsͺ *
 
ωBφ
E__inference_dense_83_layer_call_and_return_conditional_losses_1526287inputs"’
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
annotationsͺ *
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
ήBΫ
*__inference_dense_84_layer_call_fn_1526296inputs"’
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
annotationsͺ *
 
ωBφ
E__inference_dense_84_layer_call_and_return_conditional_losses_1526307inputs"’
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
annotationsͺ *
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
ήBΫ
*__inference_dense_85_layer_call_fn_1526316inputs"’
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
annotationsͺ *
 
ωBφ
E__inference_dense_85_layer_call_and_return_conditional_losses_1526327inputs"’
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
annotationsͺ *
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
ήBΫ
*__inference_dense_86_layer_call_fn_1526336inputs"’
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
annotationsͺ *
 
ωBφ
E__inference_dense_86_layer_call_and_return_conditional_losses_1526346inputs"’
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
annotationsͺ *
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
&:$^^2Adam/dense_83/kernel/m
 :^2Adam/dense_83/bias/m
&:$^/2Adam/dense_84/kernel/m
 :/2Adam/dense_84/bias/m
&:$//2Adam/dense_85/kernel/m
 :/2Adam/dense_85/bias/m
&:$/2Adam/dense_86/kernel/m
 :2Adam/dense_86/bias/m
&:$^^2Adam/dense_83/kernel/v
 :^2Adam/dense_83/bias/v
&:$^/2Adam/dense_84/kernel/v
 :/2Adam/dense_84/bias/v
&:$//2Adam/dense_85/kernel/v
 :/2Adam/dense_85/bias/v
&:$/2Adam/dense_86/kernel/v
 :2Adam/dense_86/bias/vΕ
Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1526094p#$+,34=’:
3’0
&#
input_57?????????/
p 

 
ͺ "%’"

0?????????
 Ε
Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1526119p#$+,34=’:
3’0
&#
input_57?????????/
p

 
ͺ "%’"

0?????????
 Γ
Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1526223n#$+,34;’8
1’.
$!
inputs?????????/
p 

 
ͺ "%’"

0?????????
 Γ
Q__inference_Flatten3Layers_Mixed_layer_call_and_return_conditional_losses_1526256n#$+,34;’8
1’.
$!
inputs?????????/
p

 
ͺ "%’"

0?????????
 
6__inference_Flatten3Layers_Mixed_layer_call_fn_1525935c#$+,34=’:
3’0
&#
input_57?????????/
p 

 
ͺ "?????????
6__inference_Flatten3Layers_Mixed_layer_call_fn_1526069c#$+,34=’:
3’0
&#
input_57?????????/
p

 
ͺ "?????????
6__inference_Flatten3Layers_Mixed_layer_call_fn_1526169a#$+,34;’8
1’.
$!
inputs?????????/
p 

 
ͺ "?????????
6__inference_Flatten3Layers_Mixed_layer_call_fn_1526190a#$+,34;’8
1’.
$!
inputs?????????/
p

 
ͺ "?????????
"__inference__wrapped_model_1525833v#$+,345’2
+’(
&#
input_57?????????/
ͺ "3ͺ0
.
dense_86"
dense_86?????????₯
E__inference_dense_83_layer_call_and_return_conditional_losses_1526287\/’,
%’"
 
inputs?????????^
ͺ "%’"

0?????????^
 }
*__inference_dense_83_layer_call_fn_1526276O/’,
%’"
 
inputs?????????^
ͺ "?????????^₯
E__inference_dense_84_layer_call_and_return_conditional_losses_1526307\#$/’,
%’"
 
inputs?????????^
ͺ "%’"

0?????????/
 }
*__inference_dense_84_layer_call_fn_1526296O#$/’,
%’"
 
inputs?????????^
ͺ "?????????/₯
E__inference_dense_85_layer_call_and_return_conditional_losses_1526327\+,/’,
%’"
 
inputs?????????/
ͺ "%’"

0?????????/
 }
*__inference_dense_85_layer_call_fn_1526316O+,/’,
%’"
 
inputs?????????/
ͺ "?????????/₯
E__inference_dense_86_layer_call_and_return_conditional_losses_1526346\34/’,
%’"
 
inputs?????????/
ͺ "%’"

0?????????
 }
*__inference_dense_86_layer_call_fn_1526336O34/’,
%’"
 
inputs?????????/
ͺ "?????????§
G__inference_flatten_11_layer_call_and_return_conditional_losses_1526267\3’0
)’&
$!
inputs?????????/
ͺ "%’"

0?????????^
 
,__inference_flatten_11_layer_call_fn_1526261O3’0
)’&
$!
inputs?????????/
ͺ "?????????^¬
%__inference_signature_wrapper_1526148#$+,34A’>
’ 
7ͺ4
2
input_57&#
input_57?????????/"3ͺ0
.
dense_86"
dense_86?????????