Þ
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
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8ßâ

Adam/dense_105/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_105/bias/v
{
)Adam/dense_105/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_105/bias/v*
_output_shapes
:*
dtype0

Adam/dense_105/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/dense_105/kernel/v

+Adam/dense_105/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_105/kernel/v*
_output_shapes

: *
dtype0

Adam/dense_104/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_104/bias/v
{
)Adam/dense_104/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_104/bias/v*
_output_shapes
: *
dtype0

Adam/dense_104/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/dense_104/kernel/v

+Adam/dense_104/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_104/kernel/v*
_output_shapes

:  *
dtype0

Adam/dense_103/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_103/bias/v
{
)Adam/dense_103/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_103/bias/v*
_output_shapes
: *
dtype0

Adam/dense_103/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/dense_103/kernel/v

+Adam/dense_103/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_103/kernel/v*
_output_shapes

:  *
dtype0

Adam/dense_102/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_102/bias/v
{
)Adam/dense_102/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_102/bias/v*
_output_shapes
: *
dtype0

Adam/dense_102/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/ *(
shared_nameAdam/dense_102/kernel/v

+Adam/dense_102/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_102/kernel/v*
_output_shapes

:/ *
dtype0

Adam/dense_101/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*&
shared_nameAdam/dense_101/bias/v
{
)Adam/dense_101/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_101/bias/v*
_output_shapes
:/*
dtype0

Adam/dense_101/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*(
shared_nameAdam/dense_101/kernel/v

+Adam/dense_101/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_101/kernel/v*
_output_shapes

://*
dtype0

Adam/dense_105/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_105/bias/m
{
)Adam/dense_105/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_105/bias/m*
_output_shapes
:*
dtype0

Adam/dense_105/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/dense_105/kernel/m

+Adam/dense_105/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_105/kernel/m*
_output_shapes

: *
dtype0

Adam/dense_104/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_104/bias/m
{
)Adam/dense_104/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_104/bias/m*
_output_shapes
: *
dtype0

Adam/dense_104/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/dense_104/kernel/m

+Adam/dense_104/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_104/kernel/m*
_output_shapes

:  *
dtype0

Adam/dense_103/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_103/bias/m
{
)Adam/dense_103/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_103/bias/m*
_output_shapes
: *
dtype0

Adam/dense_103/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/dense_103/kernel/m

+Adam/dense_103/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_103/kernel/m*
_output_shapes

:  *
dtype0

Adam/dense_102/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_102/bias/m
{
)Adam/dense_102/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_102/bias/m*
_output_shapes
: *
dtype0

Adam/dense_102/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/ *(
shared_nameAdam/dense_102/kernel/m

+Adam/dense_102/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_102/kernel/m*
_output_shapes

:/ *
dtype0

Adam/dense_101/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*&
shared_nameAdam/dense_101/bias/m
{
)Adam/dense_101/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_101/bias/m*
_output_shapes
:/*
dtype0

Adam/dense_101/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*(
shared_nameAdam/dense_101/kernel/m

+Adam/dense_101/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_101/kernel/m*
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
t
dense_105/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_105/bias
m
"dense_105/bias/Read/ReadVariableOpReadVariableOpdense_105/bias*
_output_shapes
:*
dtype0
|
dense_105/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_namedense_105/kernel
u
$dense_105/kernel/Read/ReadVariableOpReadVariableOpdense_105/kernel*
_output_shapes

: *
dtype0
t
dense_104/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_104/bias
m
"dense_104/bias/Read/ReadVariableOpReadVariableOpdense_104/bias*
_output_shapes
: *
dtype0
|
dense_104/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_104/kernel
u
$dense_104/kernel/Read/ReadVariableOpReadVariableOpdense_104/kernel*
_output_shapes

:  *
dtype0
t
dense_103/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_103/bias
m
"dense_103/bias/Read/ReadVariableOpReadVariableOpdense_103/bias*
_output_shapes
: *
dtype0
|
dense_103/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_103/kernel
u
$dense_103/kernel/Read/ReadVariableOpReadVariableOpdense_103/kernel*
_output_shapes

:  *
dtype0
t
dense_102/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_102/bias
m
"dense_102/bias/Read/ReadVariableOpReadVariableOpdense_102/bias*
_output_shapes
: *
dtype0
|
dense_102/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/ *!
shared_namedense_102/kernel
u
$dense_102/kernel/Read/ReadVariableOpReadVariableOpdense_102/kernel*
_output_shapes

:/ *
dtype0
t
dense_101/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namedense_101/bias
m
"dense_101/bias/Read/ReadVariableOpReadVariableOpdense_101/bias*
_output_shapes
:/*
dtype0
|
dense_101/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*!
shared_namedense_101/kernel
u
$dense_101/kernel/Read/ReadVariableOpReadVariableOpdense_101/kernel*
_output_shapes

://*
dtype0
q
serving_default_input_36Placeholder*"
_output_shapes
:/*
dtype0*
shape:/
é
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_36dense_101/kerneldense_101/biasdense_102/kerneldense_102/biasdense_103/kerneldense_103/biasdense_104/kerneldense_104/biasdense_105/kerneldense_105/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1379405

NoOpNoOp
ÁF
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*üE
valueòEBïE BèE

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¦
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
¦
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias*
¦
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias*
¦
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias*
J
0
1
$2
%3
,4
-5
46
57
<8
=9*
J
0
1
$2
%3
,4
-5
46
57
<8
=9*
* 
°
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_3* 
6
Gtrace_0
Htrace_1
Itrace_2
Jtrace_3* 
* 

Kiter

Lbeta_1

Mbeta_2
	Ndecay
Olearning_ratemm$m%m,m-m4m5m<m=mvv$v%v,v-v4v5v<v=v*

Pserving_default* 
* 
* 
* 

Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Vtrace_0* 

Wtrace_0* 

0
1*

0
1*
* 

Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

]trace_0* 

^trace_0* 
`Z
VARIABLE_VALUEdense_101/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_101/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 

_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

dtrace_0* 

etrace_0* 
`Z
VARIABLE_VALUEdense_102/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_102/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

,0
-1*

,0
-1*
* 

fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

ktrace_0* 

ltrace_0* 
`Z
VARIABLE_VALUEdense_103/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_103/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

40
51*

40
51*
* 

mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

rtrace_0* 

strace_0* 
`Z
VARIABLE_VALUEdense_104/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_104/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

<0
=1*

<0
=1*
* 

tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

ytrace_0* 

ztrace_0* 
`Z
VARIABLE_VALUEdense_105/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_105/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*

{0*
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
* 
* 
* 
* 
* 
* 
* 
8
|	variables
}	keras_api
	~total
	count*

~0
1*

|	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_101/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_101/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_102/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_102/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_103/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_103/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_104/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_104/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_105/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_105/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_101/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_101/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_102/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_102/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_103/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_103/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_104/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_104/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_105/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_105/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_101/kernel/Read/ReadVariableOp"dense_101/bias/Read/ReadVariableOp$dense_102/kernel/Read/ReadVariableOp"dense_102/bias/Read/ReadVariableOp$dense_103/kernel/Read/ReadVariableOp"dense_103/bias/Read/ReadVariableOp$dense_104/kernel/Read/ReadVariableOp"dense_104/bias/Read/ReadVariableOp$dense_105/kernel/Read/ReadVariableOp"dense_105/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_101/kernel/m/Read/ReadVariableOp)Adam/dense_101/bias/m/Read/ReadVariableOp+Adam/dense_102/kernel/m/Read/ReadVariableOp)Adam/dense_102/bias/m/Read/ReadVariableOp+Adam/dense_103/kernel/m/Read/ReadVariableOp)Adam/dense_103/bias/m/Read/ReadVariableOp+Adam/dense_104/kernel/m/Read/ReadVariableOp)Adam/dense_104/bias/m/Read/ReadVariableOp+Adam/dense_105/kernel/m/Read/ReadVariableOp)Adam/dense_105/bias/m/Read/ReadVariableOp+Adam/dense_101/kernel/v/Read/ReadVariableOp)Adam/dense_101/bias/v/Read/ReadVariableOp+Adam/dense_102/kernel/v/Read/ReadVariableOp)Adam/dense_102/bias/v/Read/ReadVariableOp+Adam/dense_103/kernel/v/Read/ReadVariableOp)Adam/dense_103/bias/v/Read/ReadVariableOp+Adam/dense_104/kernel/v/Read/ReadVariableOp)Adam/dense_104/bias/v/Read/ReadVariableOp+Adam/dense_105/kernel/v/Read/ReadVariableOp)Adam/dense_105/bias/v/Read/ReadVariableOpConst*2
Tin+
)2'	*
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
 __inference__traced_save_1379770

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_101/kerneldense_101/biasdense_102/kerneldense_102/biasdense_103/kerneldense_103/biasdense_104/kerneldense_104/biasdense_105/kerneldense_105/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_101/kernel/mAdam/dense_101/bias/mAdam/dense_102/kernel/mAdam/dense_102/bias/mAdam/dense_103/kernel/mAdam/dense_103/bias/mAdam/dense_104/kernel/mAdam/dense_104/bias/mAdam/dense_105/kernel/mAdam/dense_105/bias/mAdam/dense_101/kernel/vAdam/dense_101/bias/vAdam/dense_102/kernel/vAdam/dense_102/bias/vAdam/dense_103/kernel/vAdam/dense_103/bias/vAdam/dense_104/kernel/vAdam/dense_104/bias/vAdam/dense_105/kernel/vAdam/dense_105/bias/v*1
Tin*
(2&*
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
#__inference__traced_restore_1379891¹
	
÷
F__inference_dense_103_layer_call_and_return_conditional_losses_1379089

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: V
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

: w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

: 
 
_user_specified_nameinputs
ç	
÷
F__inference_dense_101_layer_call_and_return_conditional_losses_1379057

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:/r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:/G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:/X
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

:/w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:/
 
_user_specified_nameinputs
¤


=__inference_Flatten5Layers_Mixed_batch8_layer_call_fn_1379312
input_36
unknown://
	unknown_0:/
	unknown_1:/ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8:
identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallinput_36unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *a
f\RZ
X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379264f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
"
_output_shapes
:/
"
_user_specified_name
input_36

H
,__inference_flatten_27_layer_call_fn_1379534

inputs
identity©
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_27_layer_call_and_return_conditional_losses_1379044W
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes

:/"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:/:J F
"
_output_shapes
:/
 
_user_specified_nameinputs
¤


=__inference_Flatten5Layers_Mixed_batch8_layer_call_fn_1379151
input_36
unknown://
	unknown_0:/
	unknown_1:/ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8:
identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallinput_36unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *a
f\RZ
X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379128f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
"
_output_shapes
:/
"
_user_specified_name
input_36
	
÷
F__inference_dense_105_layer_call_and_return_conditional_losses_1379121

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

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:V
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

:w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

: 
 
_user_specified_nameinputs
Î

X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379342
input_36#
dense_101_1379316://
dense_101_1379318:/#
dense_102_1379321:/ 
dense_102_1379323: #
dense_103_1379326:  
dense_103_1379328: #
dense_104_1379331:  
dense_104_1379333: #
dense_105_1379336: 
dense_105_1379338:
identity¢!dense_101/StatefulPartitionedCall¢!dense_102/StatefulPartitionedCall¢!dense_103/StatefulPartitionedCall¢!dense_104/StatefulPartitionedCall¢!dense_105/StatefulPartitionedCall¶
flatten_27/PartitionedCallPartitionedCallinput_36*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_27_layer_call_and_return_conditional_losses_1379044
!dense_101/StatefulPartitionedCallStatefulPartitionedCall#flatten_27/PartitionedCall:output:0dense_101_1379316dense_101_1379318*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_101_layer_call_and_return_conditional_losses_1379057
!dense_102/StatefulPartitionedCallStatefulPartitionedCall*dense_101/StatefulPartitionedCall:output:0dense_102_1379321dense_102_1379323*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_102_layer_call_and_return_conditional_losses_1379073
!dense_103/StatefulPartitionedCallStatefulPartitionedCall*dense_102/StatefulPartitionedCall:output:0dense_103_1379326dense_103_1379328*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_103_layer_call_and_return_conditional_losses_1379089
!dense_104/StatefulPartitionedCallStatefulPartitionedCall*dense_103/StatefulPartitionedCall:output:0dense_104_1379331dense_104_1379333*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_104_layer_call_and_return_conditional_losses_1379105
!dense_105/StatefulPartitionedCallStatefulPartitionedCall*dense_104/StatefulPartitionedCall:output:0dense_105_1379336dense_105_1379338*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_105_layer_call_and_return_conditional_losses_1379121p
IdentityIdentity*dense_105/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:ú
NoOpNoOp"^dense_101/StatefulPartitionedCall"^dense_102/StatefulPartitionedCall"^dense_103/StatefulPartitionedCall"^dense_104/StatefulPartitionedCall"^dense_105/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall2F
!dense_102/StatefulPartitionedCall!dense_102/StatefulPartitionedCall2F
!dense_103/StatefulPartitionedCall!dense_103/StatefulPartitionedCall2F
!dense_104/StatefulPartitionedCall!dense_104/StatefulPartitionedCall2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall:L H
"
_output_shapes
:/
"
_user_specified_name
input_36
	
÷
F__inference_dense_102_layer_call_and_return_conditional_losses_1379579

inputs0
matmul_readvariableop_resource:/ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/ *
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: V
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

: w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:/
 
_user_specified_nameinputs
	
÷
F__inference_dense_105_layer_call_and_return_conditional_losses_1379636

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

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:V
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

:w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

: 
 
_user_specified_nameinputs



=__inference_Flatten5Layers_Mixed_batch8_layer_call_fn_1379430

inputs
unknown://
	unknown_0:/
	unknown_1:/ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8:
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *a
f\RZ
X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379128f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:/
 
_user_specified_nameinputs
	
÷
F__inference_dense_102_layer_call_and_return_conditional_losses_1379073

inputs0
matmul_readvariableop_resource:/ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/ *
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: V
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

: w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:/
 
_user_specified_nameinputs

c
G__inference_flatten_27_layer_call_and_return_conditional_losses_1379540

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ/   S
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes

:/O
IdentityIdentityReshape:output:0*
T0*
_output_shapes

:/"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:/:J F
"
_output_shapes
:/
 
_user_specified_nameinputs
­+

X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379529

inputs:
(dense_101_matmul_readvariableop_resource://7
)dense_101_biasadd_readvariableop_resource:/:
(dense_102_matmul_readvariableop_resource:/ 7
)dense_102_biasadd_readvariableop_resource: :
(dense_103_matmul_readvariableop_resource:  7
)dense_103_biasadd_readvariableop_resource: :
(dense_104_matmul_readvariableop_resource:  7
)dense_104_biasadd_readvariableop_resource: :
(dense_105_matmul_readvariableop_resource: 7
)dense_105_biasadd_readvariableop_resource:
identity¢ dense_101/BiasAdd/ReadVariableOp¢dense_101/MatMul/ReadVariableOp¢ dense_102/BiasAdd/ReadVariableOp¢dense_102/MatMul/ReadVariableOp¢ dense_103/BiasAdd/ReadVariableOp¢dense_103/MatMul/ReadVariableOp¢ dense_104/BiasAdd/ReadVariableOp¢dense_104/MatMul/ReadVariableOp¢ dense_105/BiasAdd/ReadVariableOp¢dense_105/MatMul/ReadVariableOpa
flatten_27/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ/   i
flatten_27/ReshapeReshapeinputsflatten_27/Const:output:0*
T0*
_output_shapes

:/
dense_101/MatMul/ReadVariableOpReadVariableOp(dense_101_matmul_readvariableop_resource*
_output_shapes

://*
dtype0
dense_101/MatMulMatMulflatten_27/Reshape:output:0'dense_101/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:/
 dense_101/BiasAdd/ReadVariableOpReadVariableOp)dense_101_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0
dense_101/BiasAddBiasAdddense_101/MatMul:product:0(dense_101/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:/[
dense_101/ReluReludense_101/BiasAdd:output:0*
T0*
_output_shapes

:/
dense_102/MatMul/ReadVariableOpReadVariableOp(dense_102_matmul_readvariableop_resource*
_output_shapes

:/ *
dtype0
dense_102/MatMulMatMuldense_101/Relu:activations:0'dense_102/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
 dense_102/BiasAdd/ReadVariableOpReadVariableOp)dense_102_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_102/BiasAddBiasAdddense_102/MatMul:product:0(dense_102/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_103/MatMul/ReadVariableOpReadVariableOp(dense_103_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_103/MatMulMatMuldense_102/BiasAdd:output:0'dense_103/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
 dense_103/BiasAdd/ReadVariableOpReadVariableOp)dense_103_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_103/BiasAddBiasAdddense_103/MatMul:product:0(dense_103/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_104/MatMul/ReadVariableOpReadVariableOp(dense_104_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_104/MatMulMatMuldense_103/BiasAdd:output:0'dense_104/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
 dense_104/BiasAdd/ReadVariableOpReadVariableOp)dense_104_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_104/BiasAddBiasAdddense_104/MatMul:product:0(dense_104/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_105/MatMul/ReadVariableOpReadVariableOp(dense_105_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_105/MatMulMatMuldense_104/BiasAdd:output:0'dense_105/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_105/BiasAdd/ReadVariableOpReadVariableOp)dense_105_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_105/BiasAddBiasAdddense_105/MatMul:product:0(dense_105/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:`
IdentityIdentitydense_105/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp!^dense_101/BiasAdd/ReadVariableOp ^dense_101/MatMul/ReadVariableOp!^dense_102/BiasAdd/ReadVariableOp ^dense_102/MatMul/ReadVariableOp!^dense_103/BiasAdd/ReadVariableOp ^dense_103/MatMul/ReadVariableOp!^dense_104/BiasAdd/ReadVariableOp ^dense_104/MatMul/ReadVariableOp!^dense_105/BiasAdd/ReadVariableOp ^dense_105/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 2D
 dense_101/BiasAdd/ReadVariableOp dense_101/BiasAdd/ReadVariableOp2B
dense_101/MatMul/ReadVariableOpdense_101/MatMul/ReadVariableOp2D
 dense_102/BiasAdd/ReadVariableOp dense_102/BiasAdd/ReadVariableOp2B
dense_102/MatMul/ReadVariableOpdense_102/MatMul/ReadVariableOp2D
 dense_103/BiasAdd/ReadVariableOp dense_103/BiasAdd/ReadVariableOp2B
dense_103/MatMul/ReadVariableOpdense_103/MatMul/ReadVariableOp2D
 dense_104/BiasAdd/ReadVariableOp dense_104/BiasAdd/ReadVariableOp2B
dense_104/MatMul/ReadVariableOpdense_104/MatMul/ReadVariableOp2D
 dense_105/BiasAdd/ReadVariableOp dense_105/BiasAdd/ReadVariableOp2B
dense_105/MatMul/ReadVariableOpdense_105/MatMul/ReadVariableOp:J F
"
_output_shapes
:/
 
_user_specified_nameinputs
­+

X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379492

inputs:
(dense_101_matmul_readvariableop_resource://7
)dense_101_biasadd_readvariableop_resource:/:
(dense_102_matmul_readvariableop_resource:/ 7
)dense_102_biasadd_readvariableop_resource: :
(dense_103_matmul_readvariableop_resource:  7
)dense_103_biasadd_readvariableop_resource: :
(dense_104_matmul_readvariableop_resource:  7
)dense_104_biasadd_readvariableop_resource: :
(dense_105_matmul_readvariableop_resource: 7
)dense_105_biasadd_readvariableop_resource:
identity¢ dense_101/BiasAdd/ReadVariableOp¢dense_101/MatMul/ReadVariableOp¢ dense_102/BiasAdd/ReadVariableOp¢dense_102/MatMul/ReadVariableOp¢ dense_103/BiasAdd/ReadVariableOp¢dense_103/MatMul/ReadVariableOp¢ dense_104/BiasAdd/ReadVariableOp¢dense_104/MatMul/ReadVariableOp¢ dense_105/BiasAdd/ReadVariableOp¢dense_105/MatMul/ReadVariableOpa
flatten_27/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ/   i
flatten_27/ReshapeReshapeinputsflatten_27/Const:output:0*
T0*
_output_shapes

:/
dense_101/MatMul/ReadVariableOpReadVariableOp(dense_101_matmul_readvariableop_resource*
_output_shapes

://*
dtype0
dense_101/MatMulMatMulflatten_27/Reshape:output:0'dense_101/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:/
 dense_101/BiasAdd/ReadVariableOpReadVariableOp)dense_101_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0
dense_101/BiasAddBiasAdddense_101/MatMul:product:0(dense_101/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:/[
dense_101/ReluReludense_101/BiasAdd:output:0*
T0*
_output_shapes

:/
dense_102/MatMul/ReadVariableOpReadVariableOp(dense_102_matmul_readvariableop_resource*
_output_shapes

:/ *
dtype0
dense_102/MatMulMatMuldense_101/Relu:activations:0'dense_102/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
 dense_102/BiasAdd/ReadVariableOpReadVariableOp)dense_102_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_102/BiasAddBiasAdddense_102/MatMul:product:0(dense_102/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_103/MatMul/ReadVariableOpReadVariableOp(dense_103_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_103/MatMulMatMuldense_102/BiasAdd:output:0'dense_103/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
 dense_103/BiasAdd/ReadVariableOpReadVariableOp)dense_103_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_103/BiasAddBiasAdddense_103/MatMul:product:0(dense_103/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_104/MatMul/ReadVariableOpReadVariableOp(dense_104_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_104/MatMulMatMuldense_103/BiasAdd:output:0'dense_104/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
 dense_104/BiasAdd/ReadVariableOpReadVariableOp)dense_104_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_104/BiasAddBiasAdddense_104/MatMul:product:0(dense_104/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_105/MatMul/ReadVariableOpReadVariableOp(dense_105_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_105/MatMulMatMuldense_104/BiasAdd:output:0'dense_105/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_105/BiasAdd/ReadVariableOpReadVariableOp)dense_105_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_105/BiasAddBiasAdddense_105/MatMul:product:0(dense_105/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:`
IdentityIdentitydense_105/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp!^dense_101/BiasAdd/ReadVariableOp ^dense_101/MatMul/ReadVariableOp!^dense_102/BiasAdd/ReadVariableOp ^dense_102/MatMul/ReadVariableOp!^dense_103/BiasAdd/ReadVariableOp ^dense_103/MatMul/ReadVariableOp!^dense_104/BiasAdd/ReadVariableOp ^dense_104/MatMul/ReadVariableOp!^dense_105/BiasAdd/ReadVariableOp ^dense_105/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 2D
 dense_101/BiasAdd/ReadVariableOp dense_101/BiasAdd/ReadVariableOp2B
dense_101/MatMul/ReadVariableOpdense_101/MatMul/ReadVariableOp2D
 dense_102/BiasAdd/ReadVariableOp dense_102/BiasAdd/ReadVariableOp2B
dense_102/MatMul/ReadVariableOpdense_102/MatMul/ReadVariableOp2D
 dense_103/BiasAdd/ReadVariableOp dense_103/BiasAdd/ReadVariableOp2B
dense_103/MatMul/ReadVariableOpdense_103/MatMul/ReadVariableOp2D
 dense_104/BiasAdd/ReadVariableOp dense_104/BiasAdd/ReadVariableOp2B
dense_104/MatMul/ReadVariableOpdense_104/MatMul/ReadVariableOp2D
 dense_105/BiasAdd/ReadVariableOp dense_105/BiasAdd/ReadVariableOp2B
dense_105/MatMul/ReadVariableOpdense_105/MatMul/ReadVariableOp:J F
"
_output_shapes
:/
 
_user_specified_nameinputs



=__inference_Flatten5Layers_Mixed_batch8_layer_call_fn_1379455

inputs
unknown://
	unknown_0:/
	unknown_1:/ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8:
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *a
f\RZ
X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379264f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:/
 
_user_specified_nameinputs
¢

+__inference_dense_105_layer_call_fn_1379626

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallÒ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_105_layer_call_and_return_conditional_losses_1379121f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

: 
 
_user_specified_nameinputs
¢

+__inference_dense_101_layer_call_fn_1379549

inputs
unknown://
	unknown_0:/
identity¢StatefulPartitionedCallÒ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_101_layer_call_and_return_conditional_losses_1379057f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:/`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:/: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:/
 
_user_specified_nameinputs

c
G__inference_flatten_27_layer_call_and_return_conditional_losses_1379044

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ/   S
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes

:/O
IdentityIdentityReshape:output:0*
T0*
_output_shapes

:/"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:/:J F
"
_output_shapes
:/
 
_user_specified_nameinputs
¢

+__inference_dense_103_layer_call_fn_1379588

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallÒ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_103_layer_call_and_return_conditional_losses_1379089f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

: 
 
_user_specified_nameinputs
Ö	
ì
%__inference_signature_wrapper_1379405
input_36
unknown://
	unknown_0:/
	unknown_1:/ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_36unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_1379031f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
"
_output_shapes
:/
"
_user_specified_name
input_36
¢

+__inference_dense_104_layer_call_fn_1379607

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallÒ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_104_layer_call_and_return_conditional_losses_1379105f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

: 
 
_user_specified_nameinputs
Î

X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379372
input_36#
dense_101_1379346://
dense_101_1379348:/#
dense_102_1379351:/ 
dense_102_1379353: #
dense_103_1379356:  
dense_103_1379358: #
dense_104_1379361:  
dense_104_1379363: #
dense_105_1379366: 
dense_105_1379368:
identity¢!dense_101/StatefulPartitionedCall¢!dense_102/StatefulPartitionedCall¢!dense_103/StatefulPartitionedCall¢!dense_104/StatefulPartitionedCall¢!dense_105/StatefulPartitionedCall¶
flatten_27/PartitionedCallPartitionedCallinput_36*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_27_layer_call_and_return_conditional_losses_1379044
!dense_101/StatefulPartitionedCallStatefulPartitionedCall#flatten_27/PartitionedCall:output:0dense_101_1379346dense_101_1379348*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_101_layer_call_and_return_conditional_losses_1379057
!dense_102/StatefulPartitionedCallStatefulPartitionedCall*dense_101/StatefulPartitionedCall:output:0dense_102_1379351dense_102_1379353*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_102_layer_call_and_return_conditional_losses_1379073
!dense_103/StatefulPartitionedCallStatefulPartitionedCall*dense_102/StatefulPartitionedCall:output:0dense_103_1379356dense_103_1379358*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_103_layer_call_and_return_conditional_losses_1379089
!dense_104/StatefulPartitionedCallStatefulPartitionedCall*dense_103/StatefulPartitionedCall:output:0dense_104_1379361dense_104_1379363*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_104_layer_call_and_return_conditional_losses_1379105
!dense_105/StatefulPartitionedCallStatefulPartitionedCall*dense_104/StatefulPartitionedCall:output:0dense_105_1379366dense_105_1379368*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_105_layer_call_and_return_conditional_losses_1379121p
IdentityIdentity*dense_105/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:ú
NoOpNoOp"^dense_101/StatefulPartitionedCall"^dense_102/StatefulPartitionedCall"^dense_103/StatefulPartitionedCall"^dense_104/StatefulPartitionedCall"^dense_105/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall2F
!dense_102/StatefulPartitionedCall!dense_102/StatefulPartitionedCall2F
!dense_103/StatefulPartitionedCall!dense_103/StatefulPartitionedCall2F
!dense_104/StatefulPartitionedCall!dense_104/StatefulPartitionedCall2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall:L H
"
_output_shapes
:/
"
_user_specified_name
input_36
B

"__inference__wrapped_model_1379031
input_36V
Dflatten5layers_mixed_batch8_dense_101_matmul_readvariableop_resource://S
Eflatten5layers_mixed_batch8_dense_101_biasadd_readvariableop_resource:/V
Dflatten5layers_mixed_batch8_dense_102_matmul_readvariableop_resource:/ S
Eflatten5layers_mixed_batch8_dense_102_biasadd_readvariableop_resource: V
Dflatten5layers_mixed_batch8_dense_103_matmul_readvariableop_resource:  S
Eflatten5layers_mixed_batch8_dense_103_biasadd_readvariableop_resource: V
Dflatten5layers_mixed_batch8_dense_104_matmul_readvariableop_resource:  S
Eflatten5layers_mixed_batch8_dense_104_biasadd_readvariableop_resource: V
Dflatten5layers_mixed_batch8_dense_105_matmul_readvariableop_resource: S
Eflatten5layers_mixed_batch8_dense_105_biasadd_readvariableop_resource:
identity¢<Flatten5Layers_Mixed_batch8/dense_101/BiasAdd/ReadVariableOp¢;Flatten5Layers_Mixed_batch8/dense_101/MatMul/ReadVariableOp¢<Flatten5Layers_Mixed_batch8/dense_102/BiasAdd/ReadVariableOp¢;Flatten5Layers_Mixed_batch8/dense_102/MatMul/ReadVariableOp¢<Flatten5Layers_Mixed_batch8/dense_103/BiasAdd/ReadVariableOp¢;Flatten5Layers_Mixed_batch8/dense_103/MatMul/ReadVariableOp¢<Flatten5Layers_Mixed_batch8/dense_104/BiasAdd/ReadVariableOp¢;Flatten5Layers_Mixed_batch8/dense_104/MatMul/ReadVariableOp¢<Flatten5Layers_Mixed_batch8/dense_105/BiasAdd/ReadVariableOp¢;Flatten5Layers_Mixed_batch8/dense_105/MatMul/ReadVariableOp}
,Flatten5Layers_Mixed_batch8/flatten_27/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ/   £
.Flatten5Layers_Mixed_batch8/flatten_27/ReshapeReshapeinput_365Flatten5Layers_Mixed_batch8/flatten_27/Const:output:0*
T0*
_output_shapes

:/À
;Flatten5Layers_Mixed_batch8/dense_101/MatMul/ReadVariableOpReadVariableOpDflatten5layers_mixed_batch8_dense_101_matmul_readvariableop_resource*
_output_shapes

://*
dtype0Ý
,Flatten5Layers_Mixed_batch8/dense_101/MatMulMatMul7Flatten5Layers_Mixed_batch8/flatten_27/Reshape:output:0CFlatten5Layers_Mixed_batch8/dense_101/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:/¾
<Flatten5Layers_Mixed_batch8/dense_101/BiasAdd/ReadVariableOpReadVariableOpEflatten5layers_mixed_batch8_dense_101_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0ß
-Flatten5Layers_Mixed_batch8/dense_101/BiasAddBiasAdd6Flatten5Layers_Mixed_batch8/dense_101/MatMul:product:0DFlatten5Layers_Mixed_batch8/dense_101/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:/
*Flatten5Layers_Mixed_batch8/dense_101/ReluRelu6Flatten5Layers_Mixed_batch8/dense_101/BiasAdd:output:0*
T0*
_output_shapes

:/À
;Flatten5Layers_Mixed_batch8/dense_102/MatMul/ReadVariableOpReadVariableOpDflatten5layers_mixed_batch8_dense_102_matmul_readvariableop_resource*
_output_shapes

:/ *
dtype0Þ
,Flatten5Layers_Mixed_batch8/dense_102/MatMulMatMul8Flatten5Layers_Mixed_batch8/dense_101/Relu:activations:0CFlatten5Layers_Mixed_batch8/dense_102/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ¾
<Flatten5Layers_Mixed_batch8/dense_102/BiasAdd/ReadVariableOpReadVariableOpEflatten5layers_mixed_batch8_dense_102_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ß
-Flatten5Layers_Mixed_batch8/dense_102/BiasAddBiasAdd6Flatten5Layers_Mixed_batch8/dense_102/MatMul:product:0DFlatten5Layers_Mixed_batch8/dense_102/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: À
;Flatten5Layers_Mixed_batch8/dense_103/MatMul/ReadVariableOpReadVariableOpDflatten5layers_mixed_batch8_dense_103_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ü
,Flatten5Layers_Mixed_batch8/dense_103/MatMulMatMul6Flatten5Layers_Mixed_batch8/dense_102/BiasAdd:output:0CFlatten5Layers_Mixed_batch8/dense_103/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ¾
<Flatten5Layers_Mixed_batch8/dense_103/BiasAdd/ReadVariableOpReadVariableOpEflatten5layers_mixed_batch8_dense_103_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ß
-Flatten5Layers_Mixed_batch8/dense_103/BiasAddBiasAdd6Flatten5Layers_Mixed_batch8/dense_103/MatMul:product:0DFlatten5Layers_Mixed_batch8/dense_103/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: À
;Flatten5Layers_Mixed_batch8/dense_104/MatMul/ReadVariableOpReadVariableOpDflatten5layers_mixed_batch8_dense_104_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ü
,Flatten5Layers_Mixed_batch8/dense_104/MatMulMatMul6Flatten5Layers_Mixed_batch8/dense_103/BiasAdd:output:0CFlatten5Layers_Mixed_batch8/dense_104/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ¾
<Flatten5Layers_Mixed_batch8/dense_104/BiasAdd/ReadVariableOpReadVariableOpEflatten5layers_mixed_batch8_dense_104_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ß
-Flatten5Layers_Mixed_batch8/dense_104/BiasAddBiasAdd6Flatten5Layers_Mixed_batch8/dense_104/MatMul:product:0DFlatten5Layers_Mixed_batch8/dense_104/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: À
;Flatten5Layers_Mixed_batch8/dense_105/MatMul/ReadVariableOpReadVariableOpDflatten5layers_mixed_batch8_dense_105_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ü
,Flatten5Layers_Mixed_batch8/dense_105/MatMulMatMul6Flatten5Layers_Mixed_batch8/dense_104/BiasAdd:output:0CFlatten5Layers_Mixed_batch8/dense_105/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:¾
<Flatten5Layers_Mixed_batch8/dense_105/BiasAdd/ReadVariableOpReadVariableOpEflatten5layers_mixed_batch8_dense_105_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ß
-Flatten5Layers_Mixed_batch8/dense_105/BiasAddBiasAdd6Flatten5Layers_Mixed_batch8/dense_105/MatMul:product:0DFlatten5Layers_Mixed_batch8/dense_105/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:|
IdentityIdentity6Flatten5Layers_Mixed_batch8/dense_105/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:·
NoOpNoOp=^Flatten5Layers_Mixed_batch8/dense_101/BiasAdd/ReadVariableOp<^Flatten5Layers_Mixed_batch8/dense_101/MatMul/ReadVariableOp=^Flatten5Layers_Mixed_batch8/dense_102/BiasAdd/ReadVariableOp<^Flatten5Layers_Mixed_batch8/dense_102/MatMul/ReadVariableOp=^Flatten5Layers_Mixed_batch8/dense_103/BiasAdd/ReadVariableOp<^Flatten5Layers_Mixed_batch8/dense_103/MatMul/ReadVariableOp=^Flatten5Layers_Mixed_batch8/dense_104/BiasAdd/ReadVariableOp<^Flatten5Layers_Mixed_batch8/dense_104/MatMul/ReadVariableOp=^Flatten5Layers_Mixed_batch8/dense_105/BiasAdd/ReadVariableOp<^Flatten5Layers_Mixed_batch8/dense_105/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 2|
<Flatten5Layers_Mixed_batch8/dense_101/BiasAdd/ReadVariableOp<Flatten5Layers_Mixed_batch8/dense_101/BiasAdd/ReadVariableOp2z
;Flatten5Layers_Mixed_batch8/dense_101/MatMul/ReadVariableOp;Flatten5Layers_Mixed_batch8/dense_101/MatMul/ReadVariableOp2|
<Flatten5Layers_Mixed_batch8/dense_102/BiasAdd/ReadVariableOp<Flatten5Layers_Mixed_batch8/dense_102/BiasAdd/ReadVariableOp2z
;Flatten5Layers_Mixed_batch8/dense_102/MatMul/ReadVariableOp;Flatten5Layers_Mixed_batch8/dense_102/MatMul/ReadVariableOp2|
<Flatten5Layers_Mixed_batch8/dense_103/BiasAdd/ReadVariableOp<Flatten5Layers_Mixed_batch8/dense_103/BiasAdd/ReadVariableOp2z
;Flatten5Layers_Mixed_batch8/dense_103/MatMul/ReadVariableOp;Flatten5Layers_Mixed_batch8/dense_103/MatMul/ReadVariableOp2|
<Flatten5Layers_Mixed_batch8/dense_104/BiasAdd/ReadVariableOp<Flatten5Layers_Mixed_batch8/dense_104/BiasAdd/ReadVariableOp2z
;Flatten5Layers_Mixed_batch8/dense_104/MatMul/ReadVariableOp;Flatten5Layers_Mixed_batch8/dense_104/MatMul/ReadVariableOp2|
<Flatten5Layers_Mixed_batch8/dense_105/BiasAdd/ReadVariableOp<Flatten5Layers_Mixed_batch8/dense_105/BiasAdd/ReadVariableOp2z
;Flatten5Layers_Mixed_batch8/dense_105/MatMul/ReadVariableOp;Flatten5Layers_Mixed_batch8/dense_105/MatMul/ReadVariableOp:L H
"
_output_shapes
:/
"
_user_specified_name
input_36
N
Å
 __inference__traced_save_1379770
file_prefix/
+savev2_dense_101_kernel_read_readvariableop-
)savev2_dense_101_bias_read_readvariableop/
+savev2_dense_102_kernel_read_readvariableop-
)savev2_dense_102_bias_read_readvariableop/
+savev2_dense_103_kernel_read_readvariableop-
)savev2_dense_103_bias_read_readvariableop/
+savev2_dense_104_kernel_read_readvariableop-
)savev2_dense_104_bias_read_readvariableop/
+savev2_dense_105_kernel_read_readvariableop-
)savev2_dense_105_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_101_kernel_m_read_readvariableop4
0savev2_adam_dense_101_bias_m_read_readvariableop6
2savev2_adam_dense_102_kernel_m_read_readvariableop4
0savev2_adam_dense_102_bias_m_read_readvariableop6
2savev2_adam_dense_103_kernel_m_read_readvariableop4
0savev2_adam_dense_103_bias_m_read_readvariableop6
2savev2_adam_dense_104_kernel_m_read_readvariableop4
0savev2_adam_dense_104_bias_m_read_readvariableop6
2savev2_adam_dense_105_kernel_m_read_readvariableop4
0savev2_adam_dense_105_bias_m_read_readvariableop6
2savev2_adam_dense_101_kernel_v_read_readvariableop4
0savev2_adam_dense_101_bias_v_read_readvariableop6
2savev2_adam_dense_102_kernel_v_read_readvariableop4
0savev2_adam_dense_102_bias_v_read_readvariableop6
2savev2_adam_dense_103_kernel_v_read_readvariableop4
0savev2_adam_dense_103_bias_v_read_readvariableop6
2savev2_adam_dense_104_kernel_v_read_readvariableop4
0savev2_adam_dense_104_bias_v_read_readvariableop6
2savev2_adam_dense_105_kernel_v_read_readvariableop4
0savev2_adam_dense_105_bias_v_read_readvariableop
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
: ý
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*¦
valueB&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¹
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_101_kernel_read_readvariableop)savev2_dense_101_bias_read_readvariableop+savev2_dense_102_kernel_read_readvariableop)savev2_dense_102_bias_read_readvariableop+savev2_dense_103_kernel_read_readvariableop)savev2_dense_103_bias_read_readvariableop+savev2_dense_104_kernel_read_readvariableop)savev2_dense_104_bias_read_readvariableop+savev2_dense_105_kernel_read_readvariableop)savev2_dense_105_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_101_kernel_m_read_readvariableop0savev2_adam_dense_101_bias_m_read_readvariableop2savev2_adam_dense_102_kernel_m_read_readvariableop0savev2_adam_dense_102_bias_m_read_readvariableop2savev2_adam_dense_103_kernel_m_read_readvariableop0savev2_adam_dense_103_bias_m_read_readvariableop2savev2_adam_dense_104_kernel_m_read_readvariableop0savev2_adam_dense_104_bias_m_read_readvariableop2savev2_adam_dense_105_kernel_m_read_readvariableop0savev2_adam_dense_105_bias_m_read_readvariableop2savev2_adam_dense_101_kernel_v_read_readvariableop0savev2_adam_dense_101_bias_v_read_readvariableop2savev2_adam_dense_102_kernel_v_read_readvariableop0savev2_adam_dense_102_bias_v_read_readvariableop2savev2_adam_dense_103_kernel_v_read_readvariableop0savev2_adam_dense_103_bias_v_read_readvariableop2savev2_adam_dense_104_kernel_v_read_readvariableop0savev2_adam_dense_104_bias_v_read_readvariableop2savev2_adam_dense_105_kernel_v_read_readvariableop0savev2_adam_dense_105_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *4
dtypes*
(2&	
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

identity_1Identity_1:output:0*
_input_shapes
: ://:/:/ : :  : :  : : :: : : : : : : ://:/:/ : :  : :  : : :://:/:/ : :  : :  : : :: 2(
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

:  : 

_output_shapes
: :$	 

_output_shapes

: : 


_output_shapes
::
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
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

://: 

_output_shapes
:/:$ 

_output_shapes

:/ : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

://: 

_output_shapes
:/:$ 

_output_shapes

:/ : 

_output_shapes
: :$  

_output_shapes

:  : !

_output_shapes
: :$" 

_output_shapes

:  : #

_output_shapes
: :$$ 

_output_shapes

: : %

_output_shapes
::&

_output_shapes
: 
	
÷
F__inference_dense_104_layer_call_and_return_conditional_losses_1379617

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: V
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

: w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

: 
 
_user_specified_nameinputs
ç	
÷
F__inference_dense_101_layer_call_and_return_conditional_losses_1379560

inputs0
matmul_readvariableop_resource://-
biasadd_readvariableop_resource:/
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

://*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:/r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:/G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:/X
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

:/w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:/
 
_user_specified_nameinputs
È

X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379128

inputs#
dense_101_1379058://
dense_101_1379060:/#
dense_102_1379074:/ 
dense_102_1379076: #
dense_103_1379090:  
dense_103_1379092: #
dense_104_1379106:  
dense_104_1379108: #
dense_105_1379122: 
dense_105_1379124:
identity¢!dense_101/StatefulPartitionedCall¢!dense_102/StatefulPartitionedCall¢!dense_103/StatefulPartitionedCall¢!dense_104/StatefulPartitionedCall¢!dense_105/StatefulPartitionedCall´
flatten_27/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_27_layer_call_and_return_conditional_losses_1379044
!dense_101/StatefulPartitionedCallStatefulPartitionedCall#flatten_27/PartitionedCall:output:0dense_101_1379058dense_101_1379060*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_101_layer_call_and_return_conditional_losses_1379057
!dense_102/StatefulPartitionedCallStatefulPartitionedCall*dense_101/StatefulPartitionedCall:output:0dense_102_1379074dense_102_1379076*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_102_layer_call_and_return_conditional_losses_1379073
!dense_103/StatefulPartitionedCallStatefulPartitionedCall*dense_102/StatefulPartitionedCall:output:0dense_103_1379090dense_103_1379092*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_103_layer_call_and_return_conditional_losses_1379089
!dense_104/StatefulPartitionedCallStatefulPartitionedCall*dense_103/StatefulPartitionedCall:output:0dense_104_1379106dense_104_1379108*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_104_layer_call_and_return_conditional_losses_1379105
!dense_105/StatefulPartitionedCallStatefulPartitionedCall*dense_104/StatefulPartitionedCall:output:0dense_105_1379122dense_105_1379124*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_105_layer_call_and_return_conditional_losses_1379121p
IdentityIdentity*dense_105/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:ú
NoOpNoOp"^dense_101/StatefulPartitionedCall"^dense_102/StatefulPartitionedCall"^dense_103/StatefulPartitionedCall"^dense_104/StatefulPartitionedCall"^dense_105/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall2F
!dense_102/StatefulPartitionedCall!dense_102/StatefulPartitionedCall2F
!dense_103/StatefulPartitionedCall!dense_103/StatefulPartitionedCall2F
!dense_104/StatefulPartitionedCall!dense_104/StatefulPartitionedCall2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall:J F
"
_output_shapes
:/
 
_user_specified_nameinputs
	
÷
F__inference_dense_103_layer_call_and_return_conditional_losses_1379598

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: V
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

: w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

: 
 
_user_specified_nameinputs
	
÷
F__inference_dense_104_layer_call_and_return_conditional_losses_1379105

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: V
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

: w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

: 
 
_user_specified_nameinputs
È

X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379264

inputs#
dense_101_1379238://
dense_101_1379240:/#
dense_102_1379243:/ 
dense_102_1379245: #
dense_103_1379248:  
dense_103_1379250: #
dense_104_1379253:  
dense_104_1379255: #
dense_105_1379258: 
dense_105_1379260:
identity¢!dense_101/StatefulPartitionedCall¢!dense_102/StatefulPartitionedCall¢!dense_103/StatefulPartitionedCall¢!dense_104/StatefulPartitionedCall¢!dense_105/StatefulPartitionedCall´
flatten_27/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_27_layer_call_and_return_conditional_losses_1379044
!dense_101/StatefulPartitionedCallStatefulPartitionedCall#flatten_27/PartitionedCall:output:0dense_101_1379238dense_101_1379240*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_101_layer_call_and_return_conditional_losses_1379057
!dense_102/StatefulPartitionedCallStatefulPartitionedCall*dense_101/StatefulPartitionedCall:output:0dense_102_1379243dense_102_1379245*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_102_layer_call_and_return_conditional_losses_1379073
!dense_103/StatefulPartitionedCallStatefulPartitionedCall*dense_102/StatefulPartitionedCall:output:0dense_103_1379248dense_103_1379250*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_103_layer_call_and_return_conditional_losses_1379089
!dense_104/StatefulPartitionedCallStatefulPartitionedCall*dense_103/StatefulPartitionedCall:output:0dense_104_1379253dense_104_1379255*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_104_layer_call_and_return_conditional_losses_1379105
!dense_105/StatefulPartitionedCallStatefulPartitionedCall*dense_104/StatefulPartitionedCall:output:0dense_105_1379258dense_105_1379260*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_105_layer_call_and_return_conditional_losses_1379121p
IdentityIdentity*dense_105/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:ú
NoOpNoOp"^dense_101/StatefulPartitionedCall"^dense_102/StatefulPartitionedCall"^dense_103/StatefulPartitionedCall"^dense_104/StatefulPartitionedCall"^dense_105/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall2F
!dense_102/StatefulPartitionedCall!dense_102/StatefulPartitionedCall2F
!dense_103/StatefulPartitionedCall!dense_103/StatefulPartitionedCall2F
!dense_104/StatefulPartitionedCall!dense_104/StatefulPartitionedCall2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall:J F
"
_output_shapes
:/
 
_user_specified_nameinputs
ü
Ç
#__inference__traced_restore_1379891
file_prefix3
!assignvariableop_dense_101_kernel:///
!assignvariableop_1_dense_101_bias:/5
#assignvariableop_2_dense_102_kernel:/ /
!assignvariableop_3_dense_102_bias: 5
#assignvariableop_4_dense_103_kernel:  /
!assignvariableop_5_dense_103_bias: 5
#assignvariableop_6_dense_104_kernel:  /
!assignvariableop_7_dense_104_bias: 5
#assignvariableop_8_dense_105_kernel: /
!assignvariableop_9_dense_105_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: #
assignvariableop_15_total: #
assignvariableop_16_count: =
+assignvariableop_17_adam_dense_101_kernel_m://7
)assignvariableop_18_adam_dense_101_bias_m:/=
+assignvariableop_19_adam_dense_102_kernel_m:/ 7
)assignvariableop_20_adam_dense_102_bias_m: =
+assignvariableop_21_adam_dense_103_kernel_m:  7
)assignvariableop_22_adam_dense_103_bias_m: =
+assignvariableop_23_adam_dense_104_kernel_m:  7
)assignvariableop_24_adam_dense_104_bias_m: =
+assignvariableop_25_adam_dense_105_kernel_m: 7
)assignvariableop_26_adam_dense_105_bias_m:=
+assignvariableop_27_adam_dense_101_kernel_v://7
)assignvariableop_28_adam_dense_101_bias_v:/=
+assignvariableop_29_adam_dense_102_kernel_v:/ 7
)assignvariableop_30_adam_dense_102_bias_v: =
+assignvariableop_31_adam_dense_103_kernel_v:  7
)assignvariableop_32_adam_dense_103_bias_v: =
+assignvariableop_33_adam_dense_104_kernel_v:  7
)assignvariableop_34_adam_dense_104_bias_v: =
+assignvariableop_35_adam_dense_105_kernel_v: 7
)assignvariableop_36_adam_dense_105_bias_v:
identity_38¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*¦
valueB&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¼
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ß
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*®
_output_shapes
::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_dense_101_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_101_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_102_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_102_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_103_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_103_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_104_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_104_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp#assignvariableop_8_dense_105_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_105_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_dense_101_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_dense_101_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_dense_102_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_102_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_103_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_103_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_104_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_104_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_dense_105_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_105_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_101_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_101_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_102_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_102_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_103_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_103_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_dense_104_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_104_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_dense_105_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_dense_105_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ý
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_38IdentityIdentity_37:output:0^NoOp_1*
T0*
_output_shapes
: ê
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_38Identity_38:output:0*_
_input_shapesN
L: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
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
¢

+__inference_dense_102_layer_call_fn_1379569

inputs
unknown:/ 
	unknown_0: 
identity¢StatefulPartitionedCallÒ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_102_layer_call_and_return_conditional_losses_1379073f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:/: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:/
 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp* 
serving_default
8
input_36,
serving_default_input_36:0/4
	dense_105'
StatefulPartitionedCall:0tensorflow/serving/predict:Ñ¢
¶
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
»
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias"
_tf_keras_layer
»
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias"
_tf_keras_layer
»
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias"
_tf_keras_layer
f
0
1
$2
%3
,4
-5
46
57
<8
=9"
trackable_list_wrapper
f
0
1
$2
%3
,4
-5
46
57
<8
=9"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
©
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_32¾
=__inference_Flatten5Layers_Mixed_batch8_layer_call_fn_1379151
=__inference_Flatten5Layers_Mixed_batch8_layer_call_fn_1379430
=__inference_Flatten5Layers_Mixed_batch8_layer_call_fn_1379455
=__inference_Flatten5Layers_Mixed_batch8_layer_call_fn_1379312¿
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
 zCtrace_0zDtrace_1zEtrace_2zFtrace_3

Gtrace_0
Htrace_1
Itrace_2
Jtrace_32ª
X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379492
X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379529
X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379342
X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379372¿
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
 zGtrace_0zHtrace_1zItrace_2zJtrace_3
ÎBË
"__inference__wrapped_model_1379031input_36"
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

Kiter

Lbeta_1

Mbeta_2
	Ndecay
Olearning_ratemm$m%m,m-m4m5m<m=mvv$v%v,v-v4v5v<v=v"
	optimizer
,
Pserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ð
Vtrace_02Ó
,__inference_flatten_27_layer_call_fn_1379534¢
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
 zVtrace_0

Wtrace_02î
G__inference_flatten_27_layer_call_and_return_conditional_losses_1379540¢
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
 zWtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ï
]trace_02Ò
+__inference_dense_101_layer_call_fn_1379549¢
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
 z]trace_0

^trace_02í
F__inference_dense_101_layer_call_and_return_conditional_losses_1379560¢
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
 z^trace_0
": //2dense_101/kernel
:/2dense_101/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
ï
dtrace_02Ò
+__inference_dense_102_layer_call_fn_1379569¢
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
 zdtrace_0

etrace_02í
F__inference_dense_102_layer_call_and_return_conditional_losses_1379579¢
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
 zetrace_0
": / 2dense_102/kernel
: 2dense_102/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
ï
ktrace_02Ò
+__inference_dense_103_layer_call_fn_1379588¢
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
 zktrace_0

ltrace_02í
F__inference_dense_103_layer_call_and_return_conditional_losses_1379598¢
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
 zltrace_0
":   2dense_103/kernel
: 2dense_103/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
­
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
ï
rtrace_02Ò
+__inference_dense_104_layer_call_fn_1379607¢
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
 zrtrace_0

strace_02í
F__inference_dense_104_layer_call_and_return_conditional_losses_1379617¢
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
 zstrace_0
":   2dense_104/kernel
: 2dense_104/bias
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
ï
ytrace_02Ò
+__inference_dense_105_layer_call_fn_1379626¢
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
 zytrace_0

ztrace_02í
F__inference_dense_105_layer_call_and_return_conditional_losses_1379636¢
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
 zztrace_0
":  2dense_105/kernel
:2dense_105/bias
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
'
{0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
=__inference_Flatten5Layers_Mixed_batch8_layer_call_fn_1379151input_36"¿
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
B
=__inference_Flatten5Layers_Mixed_batch8_layer_call_fn_1379430inputs"¿
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
B
=__inference_Flatten5Layers_Mixed_batch8_layer_call_fn_1379455inputs"¿
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
B
=__inference_Flatten5Layers_Mixed_batch8_layer_call_fn_1379312input_36"¿
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
©B¦
X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379492inputs"¿
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
©B¦
X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379529inputs"¿
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
«B¨
X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379342input_36"¿
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
«B¨
X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379372input_36"¿
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
%__inference_signature_wrapper_1379405input_36"
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
,__inference_flatten_27_layer_call_fn_1379534inputs"¢
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
G__inference_flatten_27_layer_call_and_return_conditional_losses_1379540inputs"¢
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
ßBÜ
+__inference_dense_101_layer_call_fn_1379549inputs"¢
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
F__inference_dense_101_layer_call_and_return_conditional_losses_1379560inputs"¢
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
ßBÜ
+__inference_dense_102_layer_call_fn_1379569inputs"¢
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
F__inference_dense_102_layer_call_and_return_conditional_losses_1379579inputs"¢
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
ßBÜ
+__inference_dense_103_layer_call_fn_1379588inputs"¢
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
F__inference_dense_103_layer_call_and_return_conditional_losses_1379598inputs"¢
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
ßBÜ
+__inference_dense_104_layer_call_fn_1379607inputs"¢
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
F__inference_dense_104_layer_call_and_return_conditional_losses_1379617inputs"¢
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
ßBÜ
+__inference_dense_105_layer_call_fn_1379626inputs"¢
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
F__inference_dense_105_layer_call_and_return_conditional_losses_1379636inputs"¢
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
|	variables
}	keras_api
	~total
	count"
_tf_keras_metric
.
~0
1"
trackable_list_wrapper
-
|	variables"
_generic_user_object
:  (2total
:  (2count
':%//2Adam/dense_101/kernel/m
!:/2Adam/dense_101/bias/m
':%/ 2Adam/dense_102/kernel/m
!: 2Adam/dense_102/bias/m
':%  2Adam/dense_103/kernel/m
!: 2Adam/dense_103/bias/m
':%  2Adam/dense_104/kernel/m
!: 2Adam/dense_104/bias/m
':% 2Adam/dense_105/kernel/m
!:2Adam/dense_105/bias/m
':%//2Adam/dense_101/kernel/v
!:/2Adam/dense_101/bias/v
':%/ 2Adam/dense_102/kernel/v
!: 2Adam/dense_102/bias/v
':%  2Adam/dense_103/kernel/v
!: 2Adam/dense_103/bias/v
':%  2Adam/dense_104/kernel/v
!: 2Adam/dense_104/bias/v
':% 2Adam/dense_105/kernel/v
!:2Adam/dense_105/bias/v¼
X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379342`
$%,-45<=4¢1
*¢'

input_36/
p 

 
ª "¢

0
 ¼
X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379372`
$%,-45<=4¢1
*¢'

input_36/
p

 
ª "¢

0
 º
X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379492^
$%,-45<=2¢/
(¢%

inputs/
p 

 
ª "¢

0
 º
X__inference_Flatten5Layers_Mixed_batch8_layer_call_and_return_conditional_losses_1379529^
$%,-45<=2¢/
(¢%

inputs/
p

 
ª "¢

0
 
=__inference_Flatten5Layers_Mixed_batch8_layer_call_fn_1379151S
$%,-45<=4¢1
*¢'

input_36/
p 

 
ª "
=__inference_Flatten5Layers_Mixed_batch8_layer_call_fn_1379312S
$%,-45<=4¢1
*¢'

input_36/
p

 
ª "
=__inference_Flatten5Layers_Mixed_batch8_layer_call_fn_1379430Q
$%,-45<=2¢/
(¢%

inputs/
p 

 
ª "
=__inference_Flatten5Layers_Mixed_batch8_layer_call_fn_1379455Q
$%,-45<=2¢/
(¢%

inputs/
p

 
ª "
"__inference__wrapped_model_1379031h
$%,-45<=,¢)
"¢

input_36/
ª ",ª)
'
	dense_105
	dense_105
F__inference_dense_101_layer_call_and_return_conditional_losses_1379560J&¢#
¢

inputs/
ª "¢

0/
 l
+__inference_dense_101_layer_call_fn_1379549=&¢#
¢

inputs/
ª "/
F__inference_dense_102_layer_call_and_return_conditional_losses_1379579J$%&¢#
¢

inputs/
ª "¢

0 
 l
+__inference_dense_102_layer_call_fn_1379569=$%&¢#
¢

inputs/
ª " 
F__inference_dense_103_layer_call_and_return_conditional_losses_1379598J,-&¢#
¢

inputs 
ª "¢

0 
 l
+__inference_dense_103_layer_call_fn_1379588=,-&¢#
¢

inputs 
ª " 
F__inference_dense_104_layer_call_and_return_conditional_losses_1379617J45&¢#
¢

inputs 
ª "¢

0 
 l
+__inference_dense_104_layer_call_fn_1379607=45&¢#
¢

inputs 
ª " 
F__inference_dense_105_layer_call_and_return_conditional_losses_1379636J<=&¢#
¢

inputs 
ª "¢

0
 l
+__inference_dense_105_layer_call_fn_1379626=<=&¢#
¢

inputs 
ª "
G__inference_flatten_27_layer_call_and_return_conditional_losses_1379540J*¢'
 ¢

inputs/
ª "¢

0/
 m
,__inference_flatten_27_layer_call_fn_1379534=*¢'
 ¢

inputs/
ª "/
%__inference_signature_wrapper_1379405t
$%,-45<=8¢5
¢ 
.ª+
)
input_36
input_36/",ª)
'
	dense_105
	dense_105