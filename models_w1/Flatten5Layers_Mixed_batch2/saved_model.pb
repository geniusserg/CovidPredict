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
Adam/dense_115/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_115/bias/v
{
)Adam/dense_115/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_115/bias/v*
_output_shapes
:*
dtype0

Adam/dense_115/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/dense_115/kernel/v

+Adam/dense_115/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_115/kernel/v*
_output_shapes

: *
dtype0

Adam/dense_114/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_114/bias/v
{
)Adam/dense_114/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_114/bias/v*
_output_shapes
: *
dtype0

Adam/dense_114/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/dense_114/kernel/v

+Adam/dense_114/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_114/kernel/v*
_output_shapes

:  *
dtype0

Adam/dense_113/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_113/bias/v
{
)Adam/dense_113/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_113/bias/v*
_output_shapes
: *
dtype0

Adam/dense_113/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/dense_113/kernel/v

+Adam/dense_113/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_113/kernel/v*
_output_shapes

:  *
dtype0

Adam/dense_112/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_112/bias/v
{
)Adam/dense_112/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_112/bias/v*
_output_shapes
: *
dtype0

Adam/dense_112/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/ *(
shared_nameAdam/dense_112/kernel/v

+Adam/dense_112/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_112/kernel/v*
_output_shapes

:/ *
dtype0

Adam/dense_111/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*&
shared_nameAdam/dense_111/bias/v
{
)Adam/dense_111/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_111/bias/v*
_output_shapes
:/*
dtype0

Adam/dense_111/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*(
shared_nameAdam/dense_111/kernel/v

+Adam/dense_111/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_111/kernel/v*
_output_shapes

://*
dtype0

Adam/dense_115/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_115/bias/m
{
)Adam/dense_115/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_115/bias/m*
_output_shapes
:*
dtype0

Adam/dense_115/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/dense_115/kernel/m

+Adam/dense_115/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_115/kernel/m*
_output_shapes

: *
dtype0

Adam/dense_114/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_114/bias/m
{
)Adam/dense_114/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_114/bias/m*
_output_shapes
: *
dtype0

Adam/dense_114/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/dense_114/kernel/m

+Adam/dense_114/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_114/kernel/m*
_output_shapes

:  *
dtype0

Adam/dense_113/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_113/bias/m
{
)Adam/dense_113/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_113/bias/m*
_output_shapes
: *
dtype0

Adam/dense_113/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/dense_113/kernel/m

+Adam/dense_113/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_113/kernel/m*
_output_shapes

:  *
dtype0

Adam/dense_112/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_112/bias/m
{
)Adam/dense_112/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_112/bias/m*
_output_shapes
: *
dtype0

Adam/dense_112/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/ *(
shared_nameAdam/dense_112/kernel/m

+Adam/dense_112/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_112/kernel/m*
_output_shapes

:/ *
dtype0

Adam/dense_111/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*&
shared_nameAdam/dense_111/bias/m
{
)Adam/dense_111/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_111/bias/m*
_output_shapes
:/*
dtype0

Adam/dense_111/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*(
shared_nameAdam/dense_111/kernel/m

+Adam/dense_111/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_111/kernel/m*
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
dense_115/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_115/bias
m
"dense_115/bias/Read/ReadVariableOpReadVariableOpdense_115/bias*
_output_shapes
:*
dtype0
|
dense_115/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_namedense_115/kernel
u
$dense_115/kernel/Read/ReadVariableOpReadVariableOpdense_115/kernel*
_output_shapes

: *
dtype0
t
dense_114/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_114/bias
m
"dense_114/bias/Read/ReadVariableOpReadVariableOpdense_114/bias*
_output_shapes
: *
dtype0
|
dense_114/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_114/kernel
u
$dense_114/kernel/Read/ReadVariableOpReadVariableOpdense_114/kernel*
_output_shapes

:  *
dtype0
t
dense_113/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_113/bias
m
"dense_113/bias/Read/ReadVariableOpReadVariableOpdense_113/bias*
_output_shapes
: *
dtype0
|
dense_113/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_113/kernel
u
$dense_113/kernel/Read/ReadVariableOpReadVariableOpdense_113/kernel*
_output_shapes

:  *
dtype0
t
dense_112/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_112/bias
m
"dense_112/bias/Read/ReadVariableOpReadVariableOpdense_112/bias*
_output_shapes
: *
dtype0
|
dense_112/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/ *!
shared_namedense_112/kernel
u
$dense_112/kernel/Read/ReadVariableOpReadVariableOpdense_112/kernel*
_output_shapes

:/ *
dtype0
t
dense_111/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_namedense_111/bias
m
"dense_111/bias/Read/ReadVariableOpReadVariableOpdense_111/bias*
_output_shapes
:/*
dtype0
|
dense_111/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
://*!
shared_namedense_111/kernel
u
$dense_111/kernel/Read/ReadVariableOpReadVariableOpdense_111/kernel*
_output_shapes

://*
dtype0
q
serving_default_input_38Placeholder*"
_output_shapes
:/*
dtype0*
shape:/
é
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_38dense_111/kerneldense_111/biasdense_112/kerneldense_112/biasdense_113/kerneldense_113/biasdense_114/kerneldense_114/biasdense_115/kerneldense_115/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*,
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
%__inference_signature_wrapper_1641029

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
VARIABLE_VALUEdense_111/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_111/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_112/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_112/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_113/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_113/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_114/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_114/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_115/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_115/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_111/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_111/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_112/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_112/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_113/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_113/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_114/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_114/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_115/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_115/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_111/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_111/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_112/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_112/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_113/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_113/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_114/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_114/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_115/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_115/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_111/kernel/Read/ReadVariableOp"dense_111/bias/Read/ReadVariableOp$dense_112/kernel/Read/ReadVariableOp"dense_112/bias/Read/ReadVariableOp$dense_113/kernel/Read/ReadVariableOp"dense_113/bias/Read/ReadVariableOp$dense_114/kernel/Read/ReadVariableOp"dense_114/bias/Read/ReadVariableOp$dense_115/kernel/Read/ReadVariableOp"dense_115/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_111/kernel/m/Read/ReadVariableOp)Adam/dense_111/bias/m/Read/ReadVariableOp+Adam/dense_112/kernel/m/Read/ReadVariableOp)Adam/dense_112/bias/m/Read/ReadVariableOp+Adam/dense_113/kernel/m/Read/ReadVariableOp)Adam/dense_113/bias/m/Read/ReadVariableOp+Adam/dense_114/kernel/m/Read/ReadVariableOp)Adam/dense_114/bias/m/Read/ReadVariableOp+Adam/dense_115/kernel/m/Read/ReadVariableOp)Adam/dense_115/bias/m/Read/ReadVariableOp+Adam/dense_111/kernel/v/Read/ReadVariableOp)Adam/dense_111/bias/v/Read/ReadVariableOp+Adam/dense_112/kernel/v/Read/ReadVariableOp)Adam/dense_112/bias/v/Read/ReadVariableOp+Adam/dense_113/kernel/v/Read/ReadVariableOp)Adam/dense_113/bias/v/Read/ReadVariableOp+Adam/dense_114/kernel/v/Read/ReadVariableOp)Adam/dense_114/bias/v/Read/ReadVariableOp+Adam/dense_115/kernel/v/Read/ReadVariableOp)Adam/dense_115/bias/v/Read/ReadVariableOpConst*2
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
 __inference__traced_save_1641394

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_111/kerneldense_111/biasdense_112/kerneldense_112/biasdense_113/kerneldense_113/biasdense_114/kerneldense_114/biasdense_115/kerneldense_115/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_111/kernel/mAdam/dense_111/bias/mAdam/dense_112/kernel/mAdam/dense_112/bias/mAdam/dense_113/kernel/mAdam/dense_113/bias/mAdam/dense_114/kernel/mAdam/dense_114/bias/mAdam/dense_115/kernel/mAdam/dense_115/bias/mAdam/dense_111/kernel/vAdam/dense_111/bias/vAdam/dense_112/kernel/vAdam/dense_112/bias/vAdam/dense_113/kernel/vAdam/dense_113/bias/vAdam/dense_114/kernel/vAdam/dense_114/bias/vAdam/dense_115/kernel/vAdam/dense_115/bias/v*1
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
#__inference__traced_restore_1641515¹



=__inference_Flatten5Layers_Mixed_batch2_layer_call_fn_1641054

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

:*,
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
X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1640752f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:/
 
_user_specified_nameinputs

c
G__inference_flatten_29_layer_call_and_return_conditional_losses_1641164

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

:/O
IdentityIdentityReshape:output:0*
T0*
_output_shapes

:/"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:/:J F
"
_output_shapes
:/
 
_user_specified_nameinputs
	
÷
F__inference_dense_112_layer_call_and_return_conditional_losses_1641203

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

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: V
IdentityIdentityBiasAdd:output:0^NoOp*
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
:/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:/
 
_user_specified_nameinputs
	
÷
F__inference_dense_114_layer_call_and_return_conditional_losses_1640729

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

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: V
IdentityIdentityBiasAdd:output:0^NoOp*
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
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

: 
 
_user_specified_nameinputs
	
÷
F__inference_dense_113_layer_call_and_return_conditional_losses_1640713

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

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: V
IdentityIdentityBiasAdd:output:0^NoOp*
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
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

: 
 
_user_specified_nameinputs
­+

X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1641153

inputs:
(dense_111_matmul_readvariableop_resource://7
)dense_111_biasadd_readvariableop_resource:/:
(dense_112_matmul_readvariableop_resource:/ 7
)dense_112_biasadd_readvariableop_resource: :
(dense_113_matmul_readvariableop_resource:  7
)dense_113_biasadd_readvariableop_resource: :
(dense_114_matmul_readvariableop_resource:  7
)dense_114_biasadd_readvariableop_resource: :
(dense_115_matmul_readvariableop_resource: 7
)dense_115_biasadd_readvariableop_resource:
identity¢ dense_111/BiasAdd/ReadVariableOp¢dense_111/MatMul/ReadVariableOp¢ dense_112/BiasAdd/ReadVariableOp¢dense_112/MatMul/ReadVariableOp¢ dense_113/BiasAdd/ReadVariableOp¢dense_113/MatMul/ReadVariableOp¢ dense_114/BiasAdd/ReadVariableOp¢dense_114/MatMul/ReadVariableOp¢ dense_115/BiasAdd/ReadVariableOp¢dense_115/MatMul/ReadVariableOpa
flatten_29/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ/   i
flatten_29/ReshapeReshapeinputsflatten_29/Const:output:0*
T0*
_output_shapes

:/
dense_111/MatMul/ReadVariableOpReadVariableOp(dense_111_matmul_readvariableop_resource*
_output_shapes

://*
dtype0
dense_111/MatMulMatMulflatten_29/Reshape:output:0'dense_111/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:/
 dense_111/BiasAdd/ReadVariableOpReadVariableOp)dense_111_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0
dense_111/BiasAddBiasAdddense_111/MatMul:product:0(dense_111/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:/[
dense_111/ReluReludense_111/BiasAdd:output:0*
T0*
_output_shapes

:/
dense_112/MatMul/ReadVariableOpReadVariableOp(dense_112_matmul_readvariableop_resource*
_output_shapes

:/ *
dtype0
dense_112/MatMulMatMuldense_111/Relu:activations:0'dense_112/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
 dense_112/BiasAdd/ReadVariableOpReadVariableOp)dense_112_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_112/BiasAddBiasAdddense_112/MatMul:product:0(dense_112/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_113/MatMul/ReadVariableOpReadVariableOp(dense_113_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_113/MatMulMatMuldense_112/BiasAdd:output:0'dense_113/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
 dense_113/BiasAdd/ReadVariableOpReadVariableOp)dense_113_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_113/BiasAddBiasAdddense_113/MatMul:product:0(dense_113/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_114/MatMul/ReadVariableOpReadVariableOp(dense_114_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_114/MatMulMatMuldense_113/BiasAdd:output:0'dense_114/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
 dense_114/BiasAdd/ReadVariableOpReadVariableOp)dense_114_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_114/BiasAddBiasAdddense_114/MatMul:product:0(dense_114/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_115/MatMul/ReadVariableOpReadVariableOp(dense_115_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_115/MatMulMatMuldense_114/BiasAdd:output:0'dense_115/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_115/BiasAdd/ReadVariableOpReadVariableOp)dense_115_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_115/BiasAddBiasAdddense_115/MatMul:product:0(dense_115/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:`
IdentityIdentitydense_115/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp!^dense_111/BiasAdd/ReadVariableOp ^dense_111/MatMul/ReadVariableOp!^dense_112/BiasAdd/ReadVariableOp ^dense_112/MatMul/ReadVariableOp!^dense_113/BiasAdd/ReadVariableOp ^dense_113/MatMul/ReadVariableOp!^dense_114/BiasAdd/ReadVariableOp ^dense_114/MatMul/ReadVariableOp!^dense_115/BiasAdd/ReadVariableOp ^dense_115/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 2D
 dense_111/BiasAdd/ReadVariableOp dense_111/BiasAdd/ReadVariableOp2B
dense_111/MatMul/ReadVariableOpdense_111/MatMul/ReadVariableOp2D
 dense_112/BiasAdd/ReadVariableOp dense_112/BiasAdd/ReadVariableOp2B
dense_112/MatMul/ReadVariableOpdense_112/MatMul/ReadVariableOp2D
 dense_113/BiasAdd/ReadVariableOp dense_113/BiasAdd/ReadVariableOp2B
dense_113/MatMul/ReadVariableOpdense_113/MatMul/ReadVariableOp2D
 dense_114/BiasAdd/ReadVariableOp dense_114/BiasAdd/ReadVariableOp2B
dense_114/MatMul/ReadVariableOpdense_114/MatMul/ReadVariableOp2D
 dense_115/BiasAdd/ReadVariableOp dense_115/BiasAdd/ReadVariableOp2B
dense_115/MatMul/ReadVariableOpdense_115/MatMul/ReadVariableOp:J F
"
_output_shapes
:/
 
_user_specified_nameinputs

c
G__inference_flatten_29_layer_call_and_return_conditional_losses_1640668

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

:/O
IdentityIdentityReshape:output:0*
T0*
_output_shapes

:/"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:/:J F
"
_output_shapes
:/
 
_user_specified_nameinputs
¢

+__inference_dense_112_layer_call_fn_1641193

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

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_112_layer_call_and_return_conditional_losses_1640697f
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
:/: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:/
 
_user_specified_nameinputs
	
÷
F__inference_dense_115_layer_call_and_return_conditional_losses_1641260

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
¢

+__inference_dense_113_layer_call_fn_1641212

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

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_113_layer_call_and_return_conditional_losses_1640713f
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
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

: 
 
_user_specified_nameinputs

H
,__inference_flatten_29_layer_call_fn_1641158

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

:/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_29_layer_call_and_return_conditional_losses_1640668W
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes

:/"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:/:J F
"
_output_shapes
:/
 
_user_specified_nameinputs
¤


=__inference_Flatten5Layers_Mixed_batch2_layer_call_fn_1640936
input_38
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
StatefulPartitionedCallStatefulPartitionedCallinput_38unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*,
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
X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1640888f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
"
_output_shapes
:/
"
_user_specified_name
input_38
Î

X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1640996
input_38#
dense_111_1640970://
dense_111_1640972:/#
dense_112_1640975:/ 
dense_112_1640977: #
dense_113_1640980:  
dense_113_1640982: #
dense_114_1640985:  
dense_114_1640987: #
dense_115_1640990: 
dense_115_1640992:
identity¢!dense_111/StatefulPartitionedCall¢!dense_112/StatefulPartitionedCall¢!dense_113/StatefulPartitionedCall¢!dense_114/StatefulPartitionedCall¢!dense_115/StatefulPartitionedCall¶
flatten_29/PartitionedCallPartitionedCallinput_38*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_29_layer_call_and_return_conditional_losses_1640668
!dense_111/StatefulPartitionedCallStatefulPartitionedCall#flatten_29/PartitionedCall:output:0dense_111_1640970dense_111_1640972*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_111_layer_call_and_return_conditional_losses_1640681
!dense_112/StatefulPartitionedCallStatefulPartitionedCall*dense_111/StatefulPartitionedCall:output:0dense_112_1640975dense_112_1640977*
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
GPU 2J 8 *O
fJRH
F__inference_dense_112_layer_call_and_return_conditional_losses_1640697
!dense_113/StatefulPartitionedCallStatefulPartitionedCall*dense_112/StatefulPartitionedCall:output:0dense_113_1640980dense_113_1640982*
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
GPU 2J 8 *O
fJRH
F__inference_dense_113_layer_call_and_return_conditional_losses_1640713
!dense_114/StatefulPartitionedCallStatefulPartitionedCall*dense_113/StatefulPartitionedCall:output:0dense_114_1640985dense_114_1640987*
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
GPU 2J 8 *O
fJRH
F__inference_dense_114_layer_call_and_return_conditional_losses_1640729
!dense_115/StatefulPartitionedCallStatefulPartitionedCall*dense_114/StatefulPartitionedCall:output:0dense_115_1640990dense_115_1640992*
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
GPU 2J 8 *O
fJRH
F__inference_dense_115_layer_call_and_return_conditional_losses_1640745p
IdentityIdentity*dense_115/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:ú
NoOpNoOp"^dense_111/StatefulPartitionedCall"^dense_112/StatefulPartitionedCall"^dense_113/StatefulPartitionedCall"^dense_114/StatefulPartitionedCall"^dense_115/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 2F
!dense_111/StatefulPartitionedCall!dense_111/StatefulPartitionedCall2F
!dense_112/StatefulPartitionedCall!dense_112/StatefulPartitionedCall2F
!dense_113/StatefulPartitionedCall!dense_113/StatefulPartitionedCall2F
!dense_114/StatefulPartitionedCall!dense_114/StatefulPartitionedCall2F
!dense_115/StatefulPartitionedCall!dense_115/StatefulPartitionedCall:L H
"
_output_shapes
:/
"
_user_specified_name
input_38
	
÷
F__inference_dense_114_layer_call_and_return_conditional_losses_1641241

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

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: V
IdentityIdentityBiasAdd:output:0^NoOp*
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
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

: 
 
_user_specified_nameinputs



=__inference_Flatten5Layers_Mixed_batch2_layer_call_fn_1641079

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

:*,
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
X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1640888f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:/
 
_user_specified_nameinputs
N
Å
 __inference__traced_save_1641394
file_prefix/
+savev2_dense_111_kernel_read_readvariableop-
)savev2_dense_111_bias_read_readvariableop/
+savev2_dense_112_kernel_read_readvariableop-
)savev2_dense_112_bias_read_readvariableop/
+savev2_dense_113_kernel_read_readvariableop-
)savev2_dense_113_bias_read_readvariableop/
+savev2_dense_114_kernel_read_readvariableop-
)savev2_dense_114_bias_read_readvariableop/
+savev2_dense_115_kernel_read_readvariableop-
)savev2_dense_115_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_111_kernel_m_read_readvariableop4
0savev2_adam_dense_111_bias_m_read_readvariableop6
2savev2_adam_dense_112_kernel_m_read_readvariableop4
0savev2_adam_dense_112_bias_m_read_readvariableop6
2savev2_adam_dense_113_kernel_m_read_readvariableop4
0savev2_adam_dense_113_bias_m_read_readvariableop6
2savev2_adam_dense_114_kernel_m_read_readvariableop4
0savev2_adam_dense_114_bias_m_read_readvariableop6
2savev2_adam_dense_115_kernel_m_read_readvariableop4
0savev2_adam_dense_115_bias_m_read_readvariableop6
2savev2_adam_dense_111_kernel_v_read_readvariableop4
0savev2_adam_dense_111_bias_v_read_readvariableop6
2savev2_adam_dense_112_kernel_v_read_readvariableop4
0savev2_adam_dense_112_bias_v_read_readvariableop6
2savev2_adam_dense_113_kernel_v_read_readvariableop4
0savev2_adam_dense_113_bias_v_read_readvariableop6
2savev2_adam_dense_114_kernel_v_read_readvariableop4
0savev2_adam_dense_114_bias_v_read_readvariableop6
2savev2_adam_dense_115_kernel_v_read_readvariableop4
0savev2_adam_dense_115_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_111_kernel_read_readvariableop)savev2_dense_111_bias_read_readvariableop+savev2_dense_112_kernel_read_readvariableop)savev2_dense_112_bias_read_readvariableop+savev2_dense_113_kernel_read_readvariableop)savev2_dense_113_bias_read_readvariableop+savev2_dense_114_kernel_read_readvariableop)savev2_dense_114_bias_read_readvariableop+savev2_dense_115_kernel_read_readvariableop)savev2_dense_115_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_111_kernel_m_read_readvariableop0savev2_adam_dense_111_bias_m_read_readvariableop2savev2_adam_dense_112_kernel_m_read_readvariableop0savev2_adam_dense_112_bias_m_read_readvariableop2savev2_adam_dense_113_kernel_m_read_readvariableop0savev2_adam_dense_113_bias_m_read_readvariableop2savev2_adam_dense_114_kernel_m_read_readvariableop0savev2_adam_dense_114_bias_m_read_readvariableop2savev2_adam_dense_115_kernel_m_read_readvariableop0savev2_adam_dense_115_bias_m_read_readvariableop2savev2_adam_dense_111_kernel_v_read_readvariableop0savev2_adam_dense_111_bias_v_read_readvariableop2savev2_adam_dense_112_kernel_v_read_readvariableop0savev2_adam_dense_112_bias_v_read_readvariableop2savev2_adam_dense_113_kernel_v_read_readvariableop0savev2_adam_dense_113_bias_v_read_readvariableop2savev2_adam_dense_114_kernel_v_read_readvariableop0savev2_adam_dense_114_bias_v_read_readvariableop2savev2_adam_dense_115_kernel_v_read_readvariableop0savev2_adam_dense_115_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
È

X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1640888

inputs#
dense_111_1640862://
dense_111_1640864:/#
dense_112_1640867:/ 
dense_112_1640869: #
dense_113_1640872:  
dense_113_1640874: #
dense_114_1640877:  
dense_114_1640879: #
dense_115_1640882: 
dense_115_1640884:
identity¢!dense_111/StatefulPartitionedCall¢!dense_112/StatefulPartitionedCall¢!dense_113/StatefulPartitionedCall¢!dense_114/StatefulPartitionedCall¢!dense_115/StatefulPartitionedCall´
flatten_29/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_29_layer_call_and_return_conditional_losses_1640668
!dense_111/StatefulPartitionedCallStatefulPartitionedCall#flatten_29/PartitionedCall:output:0dense_111_1640862dense_111_1640864*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_111_layer_call_and_return_conditional_losses_1640681
!dense_112/StatefulPartitionedCallStatefulPartitionedCall*dense_111/StatefulPartitionedCall:output:0dense_112_1640867dense_112_1640869*
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
GPU 2J 8 *O
fJRH
F__inference_dense_112_layer_call_and_return_conditional_losses_1640697
!dense_113/StatefulPartitionedCallStatefulPartitionedCall*dense_112/StatefulPartitionedCall:output:0dense_113_1640872dense_113_1640874*
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
GPU 2J 8 *O
fJRH
F__inference_dense_113_layer_call_and_return_conditional_losses_1640713
!dense_114/StatefulPartitionedCallStatefulPartitionedCall*dense_113/StatefulPartitionedCall:output:0dense_114_1640877dense_114_1640879*
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
GPU 2J 8 *O
fJRH
F__inference_dense_114_layer_call_and_return_conditional_losses_1640729
!dense_115/StatefulPartitionedCallStatefulPartitionedCall*dense_114/StatefulPartitionedCall:output:0dense_115_1640882dense_115_1640884*
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
GPU 2J 8 *O
fJRH
F__inference_dense_115_layer_call_and_return_conditional_losses_1640745p
IdentityIdentity*dense_115/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:ú
NoOpNoOp"^dense_111/StatefulPartitionedCall"^dense_112/StatefulPartitionedCall"^dense_113/StatefulPartitionedCall"^dense_114/StatefulPartitionedCall"^dense_115/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 2F
!dense_111/StatefulPartitionedCall!dense_111/StatefulPartitionedCall2F
!dense_112/StatefulPartitionedCall!dense_112/StatefulPartitionedCall2F
!dense_113/StatefulPartitionedCall!dense_113/StatefulPartitionedCall2F
!dense_114/StatefulPartitionedCall!dense_114/StatefulPartitionedCall2F
!dense_115/StatefulPartitionedCall!dense_115/StatefulPartitionedCall:J F
"
_output_shapes
:/
 
_user_specified_nameinputs
¢

+__inference_dense_114_layer_call_fn_1641231

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

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_114_layer_call_and_return_conditional_losses_1640729f
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
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

: 
 
_user_specified_nameinputs
ç	
÷
F__inference_dense_111_layer_call_and_return_conditional_losses_1640681

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

:/r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:/G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:/X
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

:/w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:/
 
_user_specified_nameinputs
ç	
÷
F__inference_dense_111_layer_call_and_return_conditional_losses_1641184

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

:/r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:/G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:/X
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

:/w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:/
 
_user_specified_nameinputs
	
÷
F__inference_dense_113_layer_call_and_return_conditional_losses_1641222

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

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: V
IdentityIdentityBiasAdd:output:0^NoOp*
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
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

: 
 
_user_specified_nameinputs
­+

X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1641116

inputs:
(dense_111_matmul_readvariableop_resource://7
)dense_111_biasadd_readvariableop_resource:/:
(dense_112_matmul_readvariableop_resource:/ 7
)dense_112_biasadd_readvariableop_resource: :
(dense_113_matmul_readvariableop_resource:  7
)dense_113_biasadd_readvariableop_resource: :
(dense_114_matmul_readvariableop_resource:  7
)dense_114_biasadd_readvariableop_resource: :
(dense_115_matmul_readvariableop_resource: 7
)dense_115_biasadd_readvariableop_resource:
identity¢ dense_111/BiasAdd/ReadVariableOp¢dense_111/MatMul/ReadVariableOp¢ dense_112/BiasAdd/ReadVariableOp¢dense_112/MatMul/ReadVariableOp¢ dense_113/BiasAdd/ReadVariableOp¢dense_113/MatMul/ReadVariableOp¢ dense_114/BiasAdd/ReadVariableOp¢dense_114/MatMul/ReadVariableOp¢ dense_115/BiasAdd/ReadVariableOp¢dense_115/MatMul/ReadVariableOpa
flatten_29/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ/   i
flatten_29/ReshapeReshapeinputsflatten_29/Const:output:0*
T0*
_output_shapes

:/
dense_111/MatMul/ReadVariableOpReadVariableOp(dense_111_matmul_readvariableop_resource*
_output_shapes

://*
dtype0
dense_111/MatMulMatMulflatten_29/Reshape:output:0'dense_111/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:/
 dense_111/BiasAdd/ReadVariableOpReadVariableOp)dense_111_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0
dense_111/BiasAddBiasAdddense_111/MatMul:product:0(dense_111/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:/[
dense_111/ReluReludense_111/BiasAdd:output:0*
T0*
_output_shapes

:/
dense_112/MatMul/ReadVariableOpReadVariableOp(dense_112_matmul_readvariableop_resource*
_output_shapes

:/ *
dtype0
dense_112/MatMulMatMuldense_111/Relu:activations:0'dense_112/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
 dense_112/BiasAdd/ReadVariableOpReadVariableOp)dense_112_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_112/BiasAddBiasAdddense_112/MatMul:product:0(dense_112/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_113/MatMul/ReadVariableOpReadVariableOp(dense_113_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_113/MatMulMatMuldense_112/BiasAdd:output:0'dense_113/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
 dense_113/BiasAdd/ReadVariableOpReadVariableOp)dense_113_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_113/BiasAddBiasAdddense_113/MatMul:product:0(dense_113/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_114/MatMul/ReadVariableOpReadVariableOp(dense_114_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_114/MatMulMatMuldense_113/BiasAdd:output:0'dense_114/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
 dense_114/BiasAdd/ReadVariableOpReadVariableOp)dense_114_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_114/BiasAddBiasAdddense_114/MatMul:product:0(dense_114/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_115/MatMul/ReadVariableOpReadVariableOp(dense_115_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_115/MatMulMatMuldense_114/BiasAdd:output:0'dense_115/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_115/BiasAdd/ReadVariableOpReadVariableOp)dense_115_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_115/BiasAddBiasAdddense_115/MatMul:product:0(dense_115/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:`
IdentityIdentitydense_115/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp!^dense_111/BiasAdd/ReadVariableOp ^dense_111/MatMul/ReadVariableOp!^dense_112/BiasAdd/ReadVariableOp ^dense_112/MatMul/ReadVariableOp!^dense_113/BiasAdd/ReadVariableOp ^dense_113/MatMul/ReadVariableOp!^dense_114/BiasAdd/ReadVariableOp ^dense_114/MatMul/ReadVariableOp!^dense_115/BiasAdd/ReadVariableOp ^dense_115/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 2D
 dense_111/BiasAdd/ReadVariableOp dense_111/BiasAdd/ReadVariableOp2B
dense_111/MatMul/ReadVariableOpdense_111/MatMul/ReadVariableOp2D
 dense_112/BiasAdd/ReadVariableOp dense_112/BiasAdd/ReadVariableOp2B
dense_112/MatMul/ReadVariableOpdense_112/MatMul/ReadVariableOp2D
 dense_113/BiasAdd/ReadVariableOp dense_113/BiasAdd/ReadVariableOp2B
dense_113/MatMul/ReadVariableOpdense_113/MatMul/ReadVariableOp2D
 dense_114/BiasAdd/ReadVariableOp dense_114/BiasAdd/ReadVariableOp2B
dense_114/MatMul/ReadVariableOpdense_114/MatMul/ReadVariableOp2D
 dense_115/BiasAdd/ReadVariableOp dense_115/BiasAdd/ReadVariableOp2B
dense_115/MatMul/ReadVariableOpdense_115/MatMul/ReadVariableOp:J F
"
_output_shapes
:/
 
_user_specified_nameinputs
	
÷
F__inference_dense_112_layer_call_and_return_conditional_losses_1640697

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

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: V
IdentityIdentityBiasAdd:output:0^NoOp*
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
:/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:/
 
_user_specified_nameinputs
¤


=__inference_Flatten5Layers_Mixed_batch2_layer_call_fn_1640775
input_38
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
StatefulPartitionedCallStatefulPartitionedCallinput_38unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*,
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
X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1640752f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
"
_output_shapes
:/
"
_user_specified_name
input_38
	
÷
F__inference_dense_115_layer_call_and_return_conditional_losses_1640745

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
ü
Ç
#__inference__traced_restore_1641515
file_prefix3
!assignvariableop_dense_111_kernel:///
!assignvariableop_1_dense_111_bias:/5
#assignvariableop_2_dense_112_kernel:/ /
!assignvariableop_3_dense_112_bias: 5
#assignvariableop_4_dense_113_kernel:  /
!assignvariableop_5_dense_113_bias: 5
#assignvariableop_6_dense_114_kernel:  /
!assignvariableop_7_dense_114_bias: 5
#assignvariableop_8_dense_115_kernel: /
!assignvariableop_9_dense_115_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: #
assignvariableop_15_total: #
assignvariableop_16_count: =
+assignvariableop_17_adam_dense_111_kernel_m://7
)assignvariableop_18_adam_dense_111_bias_m:/=
+assignvariableop_19_adam_dense_112_kernel_m:/ 7
)assignvariableop_20_adam_dense_112_bias_m: =
+assignvariableop_21_adam_dense_113_kernel_m:  7
)assignvariableop_22_adam_dense_113_bias_m: =
+assignvariableop_23_adam_dense_114_kernel_m:  7
)assignvariableop_24_adam_dense_114_bias_m: =
+assignvariableop_25_adam_dense_115_kernel_m: 7
)assignvariableop_26_adam_dense_115_bias_m:=
+assignvariableop_27_adam_dense_111_kernel_v://7
)assignvariableop_28_adam_dense_111_bias_v:/=
+assignvariableop_29_adam_dense_112_kernel_v:/ 7
)assignvariableop_30_adam_dense_112_bias_v: =
+assignvariableop_31_adam_dense_113_kernel_v:  7
)assignvariableop_32_adam_dense_113_bias_v: =
+assignvariableop_33_adam_dense_114_kernel_v:  7
)assignvariableop_34_adam_dense_114_bias_v: =
+assignvariableop_35_adam_dense_115_kernel_v: 7
)assignvariableop_36_adam_dense_115_bias_v:
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
AssignVariableOpAssignVariableOp!assignvariableop_dense_111_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_111_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_112_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_112_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_113_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_113_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_114_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_114_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp#assignvariableop_8_dense_115_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_115_biasIdentity_9:output:0"/device:CPU:0*
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
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_dense_111_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_dense_111_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_dense_112_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_112_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_113_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_113_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_114_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_114_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_dense_115_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_115_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_111_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_111_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_112_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_112_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_113_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_113_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_dense_114_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_114_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_dense_115_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_dense_115_bias_vIdentity_36:output:0"/device:CPU:0*
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
+__inference_dense_111_layer_call_fn_1641173

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

:/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_111_layer_call_and_return_conditional_losses_1640681f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:/`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:/: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:/
 
_user_specified_nameinputs
Î

X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1640966
input_38#
dense_111_1640940://
dense_111_1640942:/#
dense_112_1640945:/ 
dense_112_1640947: #
dense_113_1640950:  
dense_113_1640952: #
dense_114_1640955:  
dense_114_1640957: #
dense_115_1640960: 
dense_115_1640962:
identity¢!dense_111/StatefulPartitionedCall¢!dense_112/StatefulPartitionedCall¢!dense_113/StatefulPartitionedCall¢!dense_114/StatefulPartitionedCall¢!dense_115/StatefulPartitionedCall¶
flatten_29/PartitionedCallPartitionedCallinput_38*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_29_layer_call_and_return_conditional_losses_1640668
!dense_111/StatefulPartitionedCallStatefulPartitionedCall#flatten_29/PartitionedCall:output:0dense_111_1640940dense_111_1640942*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_111_layer_call_and_return_conditional_losses_1640681
!dense_112/StatefulPartitionedCallStatefulPartitionedCall*dense_111/StatefulPartitionedCall:output:0dense_112_1640945dense_112_1640947*
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
GPU 2J 8 *O
fJRH
F__inference_dense_112_layer_call_and_return_conditional_losses_1640697
!dense_113/StatefulPartitionedCallStatefulPartitionedCall*dense_112/StatefulPartitionedCall:output:0dense_113_1640950dense_113_1640952*
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
GPU 2J 8 *O
fJRH
F__inference_dense_113_layer_call_and_return_conditional_losses_1640713
!dense_114/StatefulPartitionedCallStatefulPartitionedCall*dense_113/StatefulPartitionedCall:output:0dense_114_1640955dense_114_1640957*
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
GPU 2J 8 *O
fJRH
F__inference_dense_114_layer_call_and_return_conditional_losses_1640729
!dense_115/StatefulPartitionedCallStatefulPartitionedCall*dense_114/StatefulPartitionedCall:output:0dense_115_1640960dense_115_1640962*
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
GPU 2J 8 *O
fJRH
F__inference_dense_115_layer_call_and_return_conditional_losses_1640745p
IdentityIdentity*dense_115/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:ú
NoOpNoOp"^dense_111/StatefulPartitionedCall"^dense_112/StatefulPartitionedCall"^dense_113/StatefulPartitionedCall"^dense_114/StatefulPartitionedCall"^dense_115/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 2F
!dense_111/StatefulPartitionedCall!dense_111/StatefulPartitionedCall2F
!dense_112/StatefulPartitionedCall!dense_112/StatefulPartitionedCall2F
!dense_113/StatefulPartitionedCall!dense_113/StatefulPartitionedCall2F
!dense_114/StatefulPartitionedCall!dense_114/StatefulPartitionedCall2F
!dense_115/StatefulPartitionedCall!dense_115/StatefulPartitionedCall:L H
"
_output_shapes
:/
"
_user_specified_name
input_38
B

"__inference__wrapped_model_1640655
input_38V
Dflatten5layers_mixed_batch2_dense_111_matmul_readvariableop_resource://S
Eflatten5layers_mixed_batch2_dense_111_biasadd_readvariableop_resource:/V
Dflatten5layers_mixed_batch2_dense_112_matmul_readvariableop_resource:/ S
Eflatten5layers_mixed_batch2_dense_112_biasadd_readvariableop_resource: V
Dflatten5layers_mixed_batch2_dense_113_matmul_readvariableop_resource:  S
Eflatten5layers_mixed_batch2_dense_113_biasadd_readvariableop_resource: V
Dflatten5layers_mixed_batch2_dense_114_matmul_readvariableop_resource:  S
Eflatten5layers_mixed_batch2_dense_114_biasadd_readvariableop_resource: V
Dflatten5layers_mixed_batch2_dense_115_matmul_readvariableop_resource: S
Eflatten5layers_mixed_batch2_dense_115_biasadd_readvariableop_resource:
identity¢<Flatten5Layers_Mixed_batch2/dense_111/BiasAdd/ReadVariableOp¢;Flatten5Layers_Mixed_batch2/dense_111/MatMul/ReadVariableOp¢<Flatten5Layers_Mixed_batch2/dense_112/BiasAdd/ReadVariableOp¢;Flatten5Layers_Mixed_batch2/dense_112/MatMul/ReadVariableOp¢<Flatten5Layers_Mixed_batch2/dense_113/BiasAdd/ReadVariableOp¢;Flatten5Layers_Mixed_batch2/dense_113/MatMul/ReadVariableOp¢<Flatten5Layers_Mixed_batch2/dense_114/BiasAdd/ReadVariableOp¢;Flatten5Layers_Mixed_batch2/dense_114/MatMul/ReadVariableOp¢<Flatten5Layers_Mixed_batch2/dense_115/BiasAdd/ReadVariableOp¢;Flatten5Layers_Mixed_batch2/dense_115/MatMul/ReadVariableOp}
,Flatten5Layers_Mixed_batch2/flatten_29/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ/   £
.Flatten5Layers_Mixed_batch2/flatten_29/ReshapeReshapeinput_385Flatten5Layers_Mixed_batch2/flatten_29/Const:output:0*
T0*
_output_shapes

:/À
;Flatten5Layers_Mixed_batch2/dense_111/MatMul/ReadVariableOpReadVariableOpDflatten5layers_mixed_batch2_dense_111_matmul_readvariableop_resource*
_output_shapes

://*
dtype0Ý
,Flatten5Layers_Mixed_batch2/dense_111/MatMulMatMul7Flatten5Layers_Mixed_batch2/flatten_29/Reshape:output:0CFlatten5Layers_Mixed_batch2/dense_111/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:/¾
<Flatten5Layers_Mixed_batch2/dense_111/BiasAdd/ReadVariableOpReadVariableOpEflatten5layers_mixed_batch2_dense_111_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0ß
-Flatten5Layers_Mixed_batch2/dense_111/BiasAddBiasAdd6Flatten5Layers_Mixed_batch2/dense_111/MatMul:product:0DFlatten5Layers_Mixed_batch2/dense_111/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:/
*Flatten5Layers_Mixed_batch2/dense_111/ReluRelu6Flatten5Layers_Mixed_batch2/dense_111/BiasAdd:output:0*
T0*
_output_shapes

:/À
;Flatten5Layers_Mixed_batch2/dense_112/MatMul/ReadVariableOpReadVariableOpDflatten5layers_mixed_batch2_dense_112_matmul_readvariableop_resource*
_output_shapes

:/ *
dtype0Þ
,Flatten5Layers_Mixed_batch2/dense_112/MatMulMatMul8Flatten5Layers_Mixed_batch2/dense_111/Relu:activations:0CFlatten5Layers_Mixed_batch2/dense_112/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ¾
<Flatten5Layers_Mixed_batch2/dense_112/BiasAdd/ReadVariableOpReadVariableOpEflatten5layers_mixed_batch2_dense_112_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ß
-Flatten5Layers_Mixed_batch2/dense_112/BiasAddBiasAdd6Flatten5Layers_Mixed_batch2/dense_112/MatMul:product:0DFlatten5Layers_Mixed_batch2/dense_112/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: À
;Flatten5Layers_Mixed_batch2/dense_113/MatMul/ReadVariableOpReadVariableOpDflatten5layers_mixed_batch2_dense_113_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ü
,Flatten5Layers_Mixed_batch2/dense_113/MatMulMatMul6Flatten5Layers_Mixed_batch2/dense_112/BiasAdd:output:0CFlatten5Layers_Mixed_batch2/dense_113/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ¾
<Flatten5Layers_Mixed_batch2/dense_113/BiasAdd/ReadVariableOpReadVariableOpEflatten5layers_mixed_batch2_dense_113_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ß
-Flatten5Layers_Mixed_batch2/dense_113/BiasAddBiasAdd6Flatten5Layers_Mixed_batch2/dense_113/MatMul:product:0DFlatten5Layers_Mixed_batch2/dense_113/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: À
;Flatten5Layers_Mixed_batch2/dense_114/MatMul/ReadVariableOpReadVariableOpDflatten5layers_mixed_batch2_dense_114_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ü
,Flatten5Layers_Mixed_batch2/dense_114/MatMulMatMul6Flatten5Layers_Mixed_batch2/dense_113/BiasAdd:output:0CFlatten5Layers_Mixed_batch2/dense_114/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ¾
<Flatten5Layers_Mixed_batch2/dense_114/BiasAdd/ReadVariableOpReadVariableOpEflatten5layers_mixed_batch2_dense_114_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ß
-Flatten5Layers_Mixed_batch2/dense_114/BiasAddBiasAdd6Flatten5Layers_Mixed_batch2/dense_114/MatMul:product:0DFlatten5Layers_Mixed_batch2/dense_114/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: À
;Flatten5Layers_Mixed_batch2/dense_115/MatMul/ReadVariableOpReadVariableOpDflatten5layers_mixed_batch2_dense_115_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ü
,Flatten5Layers_Mixed_batch2/dense_115/MatMulMatMul6Flatten5Layers_Mixed_batch2/dense_114/BiasAdd:output:0CFlatten5Layers_Mixed_batch2/dense_115/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:¾
<Flatten5Layers_Mixed_batch2/dense_115/BiasAdd/ReadVariableOpReadVariableOpEflatten5layers_mixed_batch2_dense_115_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ß
-Flatten5Layers_Mixed_batch2/dense_115/BiasAddBiasAdd6Flatten5Layers_Mixed_batch2/dense_115/MatMul:product:0DFlatten5Layers_Mixed_batch2/dense_115/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:|
IdentityIdentity6Flatten5Layers_Mixed_batch2/dense_115/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:·
NoOpNoOp=^Flatten5Layers_Mixed_batch2/dense_111/BiasAdd/ReadVariableOp<^Flatten5Layers_Mixed_batch2/dense_111/MatMul/ReadVariableOp=^Flatten5Layers_Mixed_batch2/dense_112/BiasAdd/ReadVariableOp<^Flatten5Layers_Mixed_batch2/dense_112/MatMul/ReadVariableOp=^Flatten5Layers_Mixed_batch2/dense_113/BiasAdd/ReadVariableOp<^Flatten5Layers_Mixed_batch2/dense_113/MatMul/ReadVariableOp=^Flatten5Layers_Mixed_batch2/dense_114/BiasAdd/ReadVariableOp<^Flatten5Layers_Mixed_batch2/dense_114/MatMul/ReadVariableOp=^Flatten5Layers_Mixed_batch2/dense_115/BiasAdd/ReadVariableOp<^Flatten5Layers_Mixed_batch2/dense_115/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 2|
<Flatten5Layers_Mixed_batch2/dense_111/BiasAdd/ReadVariableOp<Flatten5Layers_Mixed_batch2/dense_111/BiasAdd/ReadVariableOp2z
;Flatten5Layers_Mixed_batch2/dense_111/MatMul/ReadVariableOp;Flatten5Layers_Mixed_batch2/dense_111/MatMul/ReadVariableOp2|
<Flatten5Layers_Mixed_batch2/dense_112/BiasAdd/ReadVariableOp<Flatten5Layers_Mixed_batch2/dense_112/BiasAdd/ReadVariableOp2z
;Flatten5Layers_Mixed_batch2/dense_112/MatMul/ReadVariableOp;Flatten5Layers_Mixed_batch2/dense_112/MatMul/ReadVariableOp2|
<Flatten5Layers_Mixed_batch2/dense_113/BiasAdd/ReadVariableOp<Flatten5Layers_Mixed_batch2/dense_113/BiasAdd/ReadVariableOp2z
;Flatten5Layers_Mixed_batch2/dense_113/MatMul/ReadVariableOp;Flatten5Layers_Mixed_batch2/dense_113/MatMul/ReadVariableOp2|
<Flatten5Layers_Mixed_batch2/dense_114/BiasAdd/ReadVariableOp<Flatten5Layers_Mixed_batch2/dense_114/BiasAdd/ReadVariableOp2z
;Flatten5Layers_Mixed_batch2/dense_114/MatMul/ReadVariableOp;Flatten5Layers_Mixed_batch2/dense_114/MatMul/ReadVariableOp2|
<Flatten5Layers_Mixed_batch2/dense_115/BiasAdd/ReadVariableOp<Flatten5Layers_Mixed_batch2/dense_115/BiasAdd/ReadVariableOp2z
;Flatten5Layers_Mixed_batch2/dense_115/MatMul/ReadVariableOp;Flatten5Layers_Mixed_batch2/dense_115/MatMul/ReadVariableOp:L H
"
_output_shapes
:/
"
_user_specified_name
input_38
¢

+__inference_dense_115_layer_call_fn_1641250

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

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_115_layer_call_and_return_conditional_losses_1640745f
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
È

X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1640752

inputs#
dense_111_1640682://
dense_111_1640684:/#
dense_112_1640698:/ 
dense_112_1640700: #
dense_113_1640714:  
dense_113_1640716: #
dense_114_1640730:  
dense_114_1640732: #
dense_115_1640746: 
dense_115_1640748:
identity¢!dense_111/StatefulPartitionedCall¢!dense_112/StatefulPartitionedCall¢!dense_113/StatefulPartitionedCall¢!dense_114/StatefulPartitionedCall¢!dense_115/StatefulPartitionedCall´
flatten_29/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:/* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_29_layer_call_and_return_conditional_losses_1640668
!dense_111/StatefulPartitionedCallStatefulPartitionedCall#flatten_29/PartitionedCall:output:0dense_111_1640682dense_111_1640684*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_111_layer_call_and_return_conditional_losses_1640681
!dense_112/StatefulPartitionedCallStatefulPartitionedCall*dense_111/StatefulPartitionedCall:output:0dense_112_1640698dense_112_1640700*
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
GPU 2J 8 *O
fJRH
F__inference_dense_112_layer_call_and_return_conditional_losses_1640697
!dense_113/StatefulPartitionedCallStatefulPartitionedCall*dense_112/StatefulPartitionedCall:output:0dense_113_1640714dense_113_1640716*
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
GPU 2J 8 *O
fJRH
F__inference_dense_113_layer_call_and_return_conditional_losses_1640713
!dense_114/StatefulPartitionedCallStatefulPartitionedCall*dense_113/StatefulPartitionedCall:output:0dense_114_1640730dense_114_1640732*
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
GPU 2J 8 *O
fJRH
F__inference_dense_114_layer_call_and_return_conditional_losses_1640729
!dense_115/StatefulPartitionedCallStatefulPartitionedCall*dense_114/StatefulPartitionedCall:output:0dense_115_1640746dense_115_1640748*
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
GPU 2J 8 *O
fJRH
F__inference_dense_115_layer_call_and_return_conditional_losses_1640745p
IdentityIdentity*dense_115/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:ú
NoOpNoOp"^dense_111/StatefulPartitionedCall"^dense_112/StatefulPartitionedCall"^dense_113/StatefulPartitionedCall"^dense_114/StatefulPartitionedCall"^dense_115/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 2F
!dense_111/StatefulPartitionedCall!dense_111/StatefulPartitionedCall2F
!dense_112/StatefulPartitionedCall!dense_112/StatefulPartitionedCall2F
!dense_113/StatefulPartitionedCall!dense_113/StatefulPartitionedCall2F
!dense_114/StatefulPartitionedCall!dense_114/StatefulPartitionedCall2F
!dense_115/StatefulPartitionedCall!dense_115/StatefulPartitionedCall:J F
"
_output_shapes
:/
 
_user_specified_nameinputs
Ö	
ì
%__inference_signature_wrapper_1641029
input_38
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
StatefulPartitionedCallStatefulPartitionedCallinput_38unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*,
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
"__inference__wrapped_model_1640655f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":/: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
"
_output_shapes
:/
"
_user_specified_name
input_38"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp* 
serving_default
8
input_38,
serving_default_input_38:0/4
	dense_115'
StatefulPartitionedCall:0tensorflow/serving/predict:Ñ¢
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
=__inference_Flatten5Layers_Mixed_batch2_layer_call_fn_1640775
=__inference_Flatten5Layers_Mixed_batch2_layer_call_fn_1641054
=__inference_Flatten5Layers_Mixed_batch2_layer_call_fn_1641079
=__inference_Flatten5Layers_Mixed_batch2_layer_call_fn_1640936¿
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
X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1641116
X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1641153
X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1640966
X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1640996¿
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
"__inference__wrapped_model_1640655input_38"
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
,__inference_flatten_29_layer_call_fn_1641158¢
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
G__inference_flatten_29_layer_call_and_return_conditional_losses_1641164¢
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
+__inference_dense_111_layer_call_fn_1641173¢
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
F__inference_dense_111_layer_call_and_return_conditional_losses_1641184¢
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
": //2dense_111/kernel
:/2dense_111/bias
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
+__inference_dense_112_layer_call_fn_1641193¢
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
F__inference_dense_112_layer_call_and_return_conditional_losses_1641203¢
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
": / 2dense_112/kernel
: 2dense_112/bias
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
+__inference_dense_113_layer_call_fn_1641212¢
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
F__inference_dense_113_layer_call_and_return_conditional_losses_1641222¢
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
":   2dense_113/kernel
: 2dense_113/bias
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
+__inference_dense_114_layer_call_fn_1641231¢
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
F__inference_dense_114_layer_call_and_return_conditional_losses_1641241¢
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
":   2dense_114/kernel
: 2dense_114/bias
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
+__inference_dense_115_layer_call_fn_1641250¢
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
F__inference_dense_115_layer_call_and_return_conditional_losses_1641260¢
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
":  2dense_115/kernel
:2dense_115/bias
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
=__inference_Flatten5Layers_Mixed_batch2_layer_call_fn_1640775input_38"¿
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
=__inference_Flatten5Layers_Mixed_batch2_layer_call_fn_1641054inputs"¿
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
=__inference_Flatten5Layers_Mixed_batch2_layer_call_fn_1641079inputs"¿
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
=__inference_Flatten5Layers_Mixed_batch2_layer_call_fn_1640936input_38"¿
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
X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1641116inputs"¿
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
X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1641153inputs"¿
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
X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1640966input_38"¿
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
X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1640996input_38"¿
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
%__inference_signature_wrapper_1641029input_38"
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
,__inference_flatten_29_layer_call_fn_1641158inputs"¢
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
G__inference_flatten_29_layer_call_and_return_conditional_losses_1641164inputs"¢
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
+__inference_dense_111_layer_call_fn_1641173inputs"¢
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
F__inference_dense_111_layer_call_and_return_conditional_losses_1641184inputs"¢
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
+__inference_dense_112_layer_call_fn_1641193inputs"¢
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
F__inference_dense_112_layer_call_and_return_conditional_losses_1641203inputs"¢
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
+__inference_dense_113_layer_call_fn_1641212inputs"¢
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
F__inference_dense_113_layer_call_and_return_conditional_losses_1641222inputs"¢
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
+__inference_dense_114_layer_call_fn_1641231inputs"¢
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
F__inference_dense_114_layer_call_and_return_conditional_losses_1641241inputs"¢
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
+__inference_dense_115_layer_call_fn_1641250inputs"¢
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
F__inference_dense_115_layer_call_and_return_conditional_losses_1641260inputs"¢
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
':%//2Adam/dense_111/kernel/m
!:/2Adam/dense_111/bias/m
':%/ 2Adam/dense_112/kernel/m
!: 2Adam/dense_112/bias/m
':%  2Adam/dense_113/kernel/m
!: 2Adam/dense_113/bias/m
':%  2Adam/dense_114/kernel/m
!: 2Adam/dense_114/bias/m
':% 2Adam/dense_115/kernel/m
!:2Adam/dense_115/bias/m
':%//2Adam/dense_111/kernel/v
!:/2Adam/dense_111/bias/v
':%/ 2Adam/dense_112/kernel/v
!: 2Adam/dense_112/bias/v
':%  2Adam/dense_113/kernel/v
!: 2Adam/dense_113/bias/v
':%  2Adam/dense_114/kernel/v
!: 2Adam/dense_114/bias/v
':% 2Adam/dense_115/kernel/v
!:2Adam/dense_115/bias/v¼
X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1640966`
$%,-45<=4¢1
*¢'

input_38/
p 

 
ª "¢

0
 ¼
X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1640996`
$%,-45<=4¢1
*¢'

input_38/
p

 
ª "¢

0
 º
X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1641116^
$%,-45<=2¢/
(¢%

inputs/
p 

 
ª "¢

0
 º
X__inference_Flatten5Layers_Mixed_batch2_layer_call_and_return_conditional_losses_1641153^
$%,-45<=2¢/
(¢%

inputs/
p

 
ª "¢

0
 
=__inference_Flatten5Layers_Mixed_batch2_layer_call_fn_1640775S
$%,-45<=4¢1
*¢'

input_38/
p 

 
ª "
=__inference_Flatten5Layers_Mixed_batch2_layer_call_fn_1640936S
$%,-45<=4¢1
*¢'

input_38/
p

 
ª "
=__inference_Flatten5Layers_Mixed_batch2_layer_call_fn_1641054Q
$%,-45<=2¢/
(¢%

inputs/
p 

 
ª "
=__inference_Flatten5Layers_Mixed_batch2_layer_call_fn_1641079Q
$%,-45<=2¢/
(¢%

inputs/
p

 
ª "
"__inference__wrapped_model_1640655h
$%,-45<=,¢)
"¢

input_38/
ª ",ª)
'
	dense_115
	dense_115
F__inference_dense_111_layer_call_and_return_conditional_losses_1641184J&¢#
¢

inputs/
ª "¢

0/
 l
+__inference_dense_111_layer_call_fn_1641173=&¢#
¢

inputs/
ª "/
F__inference_dense_112_layer_call_and_return_conditional_losses_1641203J$%&¢#
¢

inputs/
ª "¢

0 
 l
+__inference_dense_112_layer_call_fn_1641193=$%&¢#
¢

inputs/
ª " 
F__inference_dense_113_layer_call_and_return_conditional_losses_1641222J,-&¢#
¢

inputs 
ª "¢

0 
 l
+__inference_dense_113_layer_call_fn_1641212=,-&¢#
¢

inputs 
ª " 
F__inference_dense_114_layer_call_and_return_conditional_losses_1641241J45&¢#
¢

inputs 
ª "¢

0 
 l
+__inference_dense_114_layer_call_fn_1641231=45&¢#
¢

inputs 
ª " 
F__inference_dense_115_layer_call_and_return_conditional_losses_1641260J<=&¢#
¢

inputs 
ª "¢

0
 l
+__inference_dense_115_layer_call_fn_1641250=<=&¢#
¢

inputs 
ª "
G__inference_flatten_29_layer_call_and_return_conditional_losses_1641164J*¢'
 ¢

inputs/
ª "¢

0/
 m
,__inference_flatten_29_layer_call_fn_1641158=*¢'
 ¢

inputs/
ª "/
%__inference_signature_wrapper_1641029t
$%,-45<=8¢5
¢ 
.ª+
)
input_38
input_38/",ª)
'
	dense_115
	dense_115