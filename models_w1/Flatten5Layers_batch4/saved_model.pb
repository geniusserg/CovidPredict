ี
ัก
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
ม
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
executor_typestring จ
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
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8งแ
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
: *'
shared_nameAdam/dense_86/kernel/v

*Adam/dense_86/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_86/kernel/v*
_output_shapes

: *
dtype0

Adam/dense_85/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_85/bias/v
y
(Adam/dense_85/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_85/bias/v*
_output_shapes
: *
dtype0

Adam/dense_85/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_85/kernel/v

*Adam/dense_85/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_85/kernel/v*
_output_shapes

:  *
dtype0

Adam/dense_84/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_84/bias/v
y
(Adam/dense_84/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_84/bias/v*
_output_shapes
: *
dtype0

Adam/dense_84/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_84/kernel/v

*Adam/dense_84/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_84/kernel/v*
_output_shapes

:  *
dtype0

Adam/dense_83/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_83/bias/v
y
(Adam/dense_83/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_83/bias/v*
_output_shapes
: *
dtype0

Adam/dense_83/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_83/kernel/v

*Adam/dense_83/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_83/kernel/v*
_output_shapes

: *
dtype0

Adam/dense_82/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_82/bias/v
y
(Adam/dense_82/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_82/bias/v*
_output_shapes
:*
dtype0

Adam/dense_82/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_82/kernel/v

*Adam/dense_82/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_82/kernel/v*
_output_shapes

:*
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
: *'
shared_nameAdam/dense_86/kernel/m

*Adam/dense_86/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_86/kernel/m*
_output_shapes

: *
dtype0

Adam/dense_85/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_85/bias/m
y
(Adam/dense_85/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_85/bias/m*
_output_shapes
: *
dtype0

Adam/dense_85/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_85/kernel/m

*Adam/dense_85/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_85/kernel/m*
_output_shapes

:  *
dtype0

Adam/dense_84/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_84/bias/m
y
(Adam/dense_84/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_84/bias/m*
_output_shapes
: *
dtype0

Adam/dense_84/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_84/kernel/m

*Adam/dense_84/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_84/kernel/m*
_output_shapes

:  *
dtype0

Adam/dense_83/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_83/bias/m
y
(Adam/dense_83/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_83/bias/m*
_output_shapes
: *
dtype0

Adam/dense_83/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_83/kernel/m

*Adam/dense_83/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_83/kernel/m*
_output_shapes

: *
dtype0

Adam/dense_82/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_82/bias/m
y
(Adam/dense_82/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_82/bias/m*
_output_shapes
:*
dtype0

Adam/dense_82/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_82/kernel/m

*Adam/dense_82/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_82/kernel/m*
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
: * 
shared_namedense_86/kernel
s
#dense_86/kernel/Read/ReadVariableOpReadVariableOpdense_86/kernel*
_output_shapes

: *
dtype0
r
dense_85/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_85/bias
k
!dense_85/bias/Read/ReadVariableOpReadVariableOpdense_85/bias*
_output_shapes
: *
dtype0
z
dense_85/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_85/kernel
s
#dense_85/kernel/Read/ReadVariableOpReadVariableOpdense_85/kernel*
_output_shapes

:  *
dtype0
r
dense_84/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_84/bias
k
!dense_84/bias/Read/ReadVariableOpReadVariableOpdense_84/bias*
_output_shapes
: *
dtype0
z
dense_84/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_84/kernel
s
#dense_84/kernel/Read/ReadVariableOpReadVariableOpdense_84/kernel*
_output_shapes

:  *
dtype0
r
dense_83/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_83/bias
k
!dense_83/bias/Read/ReadVariableOpReadVariableOpdense_83/bias*
_output_shapes
: *
dtype0
z
dense_83/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_83/kernel
s
#dense_83/kernel/Read/ReadVariableOpReadVariableOpdense_83/kernel*
_output_shapes

: *
dtype0
r
dense_82/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_82/bias
k
!dense_82/bias/Read/ReadVariableOpReadVariableOpdense_82/bias*
_output_shapes
:*
dtype0
z
dense_82/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_82/kernel
s
#dense_82/kernel/Read/ReadVariableOpReadVariableOpdense_82/kernel*
_output_shapes

:*
dtype0
q
serving_default_input_31Placeholder*"
_output_shapes
:*
dtype0*
shape:
฿
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_31dense_82/kerneldense_82/biasdense_83/kerneldense_83/biasdense_84/kerneldense_84/biasdense_85/kerneldense_85/biasdense_86/kerneldense_86/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*,
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
%__inference_signature_wrapper_1283156

NoOpNoOp
ฃF
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?E
valueิEBัE BสE
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
ฆ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
ฆ
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
ฆ
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias*
ฆ
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias*
ฆ
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
ฐ
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
_Y
VARIABLE_VALUEdense_82/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_82/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
_Y
VARIABLE_VALUEdense_83/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_83/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
_Y
VARIABLE_VALUEdense_84/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_84/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
_Y
VARIABLE_VALUEdense_85/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_85/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
_Y
VARIABLE_VALUEdense_86/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_86/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
|
VARIABLE_VALUEAdam/dense_82/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_82/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_83/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_83/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_84/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_84/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_85/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_85/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_86/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_86/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_82/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_82/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_83/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_83/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_84/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_84/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_85/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_85/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_86/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_86/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ไ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_82/kernel/Read/ReadVariableOp!dense_82/bias/Read/ReadVariableOp#dense_83/kernel/Read/ReadVariableOp!dense_83/bias/Read/ReadVariableOp#dense_84/kernel/Read/ReadVariableOp!dense_84/bias/Read/ReadVariableOp#dense_85/kernel/Read/ReadVariableOp!dense_85/bias/Read/ReadVariableOp#dense_86/kernel/Read/ReadVariableOp!dense_86/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_82/kernel/m/Read/ReadVariableOp(Adam/dense_82/bias/m/Read/ReadVariableOp*Adam/dense_83/kernel/m/Read/ReadVariableOp(Adam/dense_83/bias/m/Read/ReadVariableOp*Adam/dense_84/kernel/m/Read/ReadVariableOp(Adam/dense_84/bias/m/Read/ReadVariableOp*Adam/dense_85/kernel/m/Read/ReadVariableOp(Adam/dense_85/bias/m/Read/ReadVariableOp*Adam/dense_86/kernel/m/Read/ReadVariableOp(Adam/dense_86/bias/m/Read/ReadVariableOp*Adam/dense_82/kernel/v/Read/ReadVariableOp(Adam/dense_82/bias/v/Read/ReadVariableOp*Adam/dense_83/kernel/v/Read/ReadVariableOp(Adam/dense_83/bias/v/Read/ReadVariableOp*Adam/dense_84/kernel/v/Read/ReadVariableOp(Adam/dense_84/bias/v/Read/ReadVariableOp*Adam/dense_85/kernel/v/Read/ReadVariableOp(Adam/dense_85/bias/v/Read/ReadVariableOp*Adam/dense_86/kernel/v/Read/ReadVariableOp(Adam/dense_86/bias/v/Read/ReadVariableOpConst*2
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
 __inference__traced_save_1283530
๛
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_82/kerneldense_82/biasdense_83/kerneldense_83/biasdense_84/kerneldense_84/biasdense_85/kerneldense_85/biasdense_86/kerneldense_86/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_82/kernel/mAdam/dense_82/bias/mAdam/dense_83/kernel/mAdam/dense_83/bias/mAdam/dense_84/kernel/mAdam/dense_84/bias/mAdam/dense_85/kernel/mAdam/dense_85/bias/mAdam/dense_86/kernel/mAdam/dense_86/bias/mAdam/dense_82/kernel/vAdam/dense_82/bias/vAdam/dense_83/kernel/vAdam/dense_83/bias/vAdam/dense_84/kernel/vAdam/dense_84/bias/vAdam/dense_85/kernel/vAdam/dense_85/bias/vAdam/dense_86/kernel/vAdam/dense_86/bias/v*1
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
#__inference__traced_restore_1283651ชน
	
๖
E__inference_dense_86_layer_call_and_return_conditional_losses_1282872

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:V
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

:w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

: 
 
_user_specified_nameinputs


?
7__inference_Flatten5Layers_batch4_layer_call_fn_1282902
input_31
unknown:
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8:
identityขStatefulPartitionedCallศ
StatefulPartitionedCallStatefulPartitionedCallinput_31unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1282879f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
"
_output_shapes
:
"
_user_specified_name
input_31


?
7__inference_Flatten5Layers_batch4_layer_call_fn_1283063
input_31
unknown:
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8:
identityขStatefulPartitionedCallศ
StatefulPartitionedCallStatefulPartitionedCallinput_31unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1283015f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
"
_output_shapes
:
"
_user_specified_name
input_31
?

*__inference_dense_85_layer_call_fn_1283366

inputs
unknown:  
	unknown_0: 
identityขStatefulPartitionedCallั
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_85_layer_call_and_return_conditional_losses_1282856f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

: 
 
_user_specified_nameinputs

๔
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1282879

inputs"
dense_82_1282806:
dense_82_1282808:"
dense_83_1282823: 
dense_83_1282825: "
dense_84_1282840:  
dense_84_1282842: "
dense_85_1282857:  
dense_85_1282859: "
dense_86_1282873: 
dense_86_1282875:
identityข dense_82/StatefulPartitionedCallข dense_83/StatefulPartitionedCallข dense_84/StatefulPartitionedCallข dense_85/StatefulPartitionedCallข dense_86/StatefulPartitionedCallด
flatten_22/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_22_layer_call_and_return_conditional_losses_1282792
 dense_82/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_82_1282806dense_82_1282808*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_82_layer_call_and_return_conditional_losses_1282805
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_1282823dense_83_1282825*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_83_layer_call_and_return_conditional_losses_1282822
 dense_84/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0dense_84_1282840dense_84_1282842*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_84_layer_call_and_return_conditional_losses_1282839
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0dense_85_1282857dense_85_1282859*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_85_layer_call_and_return_conditional_losses_1282856
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0dense_86_1282873dense_86_1282875*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_86_layer_call_and_return_conditional_losses_1282872o
IdentityIdentity)dense_86/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:๕
NoOpNoOp!^dense_82/StatefulPartitionedCall!^dense_83/StatefulPartitionedCall!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
	
๖
E__inference_dense_86_layer_call_and_return_conditional_losses_1283396

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:V
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

:w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

: 
 
_user_specified_nameinputs
ภ
ฉ
#__inference__traced_restore_1283651
file_prefix2
 assignvariableop_dense_82_kernel:.
 assignvariableop_1_dense_82_bias:4
"assignvariableop_2_dense_83_kernel: .
 assignvariableop_3_dense_83_bias: 4
"assignvariableop_4_dense_84_kernel:  .
 assignvariableop_5_dense_84_bias: 4
"assignvariableop_6_dense_85_kernel:  .
 assignvariableop_7_dense_85_bias: 4
"assignvariableop_8_dense_86_kernel: .
 assignvariableop_9_dense_86_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: #
assignvariableop_15_total: #
assignvariableop_16_count: <
*assignvariableop_17_adam_dense_82_kernel_m:6
(assignvariableop_18_adam_dense_82_bias_m:<
*assignvariableop_19_adam_dense_83_kernel_m: 6
(assignvariableop_20_adam_dense_83_bias_m: <
*assignvariableop_21_adam_dense_84_kernel_m:  6
(assignvariableop_22_adam_dense_84_bias_m: <
*assignvariableop_23_adam_dense_85_kernel_m:  6
(assignvariableop_24_adam_dense_85_bias_m: <
*assignvariableop_25_adam_dense_86_kernel_m: 6
(assignvariableop_26_adam_dense_86_bias_m:<
*assignvariableop_27_adam_dense_82_kernel_v:6
(assignvariableop_28_adam_dense_82_bias_v:<
*assignvariableop_29_adam_dense_83_kernel_v: 6
(assignvariableop_30_adam_dense_83_bias_v: <
*assignvariableop_31_adam_dense_84_kernel_v:  6
(assignvariableop_32_adam_dense_84_bias_v: <
*assignvariableop_33_adam_dense_85_kernel_v:  6
(assignvariableop_34_adam_dense_85_bias_v: <
*assignvariableop_35_adam_dense_86_kernel_v: 6
(assignvariableop_36_adam_dense_86_bias_v:
identity_38ขAssignVariableOpขAssignVariableOp_1ขAssignVariableOp_10ขAssignVariableOp_11ขAssignVariableOp_12ขAssignVariableOp_13ขAssignVariableOp_14ขAssignVariableOp_15ขAssignVariableOp_16ขAssignVariableOp_17ขAssignVariableOp_18ขAssignVariableOp_19ขAssignVariableOp_2ขAssignVariableOp_20ขAssignVariableOp_21ขAssignVariableOp_22ขAssignVariableOp_23ขAssignVariableOp_24ขAssignVariableOp_25ขAssignVariableOp_26ขAssignVariableOp_27ขAssignVariableOp_28ขAssignVariableOp_29ขAssignVariableOp_3ขAssignVariableOp_30ขAssignVariableOp_31ขAssignVariableOp_32ขAssignVariableOp_33ขAssignVariableOp_34ขAssignVariableOp_35ขAssignVariableOp_36ขAssignVariableOp_4ขAssignVariableOp_5ขAssignVariableOp_6ขAssignVariableOp_7ขAssignVariableOp_8ขAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*ฆ
valueB&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHผ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ฿
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ฎ
_output_shapes
::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_dense_82_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_82_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_83_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_83_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_84_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_84_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_85_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_85_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_86_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_86_biasIdentity_9:output:0"/device:CPU:0*
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
:
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_82_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_82_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_83_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_83_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_84_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_84_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_85_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_85_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_86_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_86_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_82_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_82_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_83_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_83_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_84_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_84_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_85_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_85_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_86_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_86_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_38IdentityIdentity_37:output:0^NoOp_1*
T0*
_output_shapes
: ๊
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
ๆ	
๖
E__inference_dense_82_layer_call_and_return_conditional_losses_1282805

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:X
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

:w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
?

*__inference_dense_86_layer_call_fn_1283386

inputs
unknown: 
	unknown_0:
identityขStatefulPartitionedCallั
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_86_layer_call_and_return_conditional_losses_1282872f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

: 
 
_user_specified_nameinputs
ุ,
?
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1283246

inputs9
'dense_82_matmul_readvariableop_resource:6
(dense_82_biasadd_readvariableop_resource:9
'dense_83_matmul_readvariableop_resource: 6
(dense_83_biasadd_readvariableop_resource: 9
'dense_84_matmul_readvariableop_resource:  6
(dense_84_biasadd_readvariableop_resource: 9
'dense_85_matmul_readvariableop_resource:  6
(dense_85_biasadd_readvariableop_resource: 9
'dense_86_matmul_readvariableop_resource: 6
(dense_86_biasadd_readvariableop_resource:
identityขdense_82/BiasAdd/ReadVariableOpขdense_82/MatMul/ReadVariableOpขdense_83/BiasAdd/ReadVariableOpขdense_83/MatMul/ReadVariableOpขdense_84/BiasAdd/ReadVariableOpขdense_84/MatMul/ReadVariableOpขdense_85/BiasAdd/ReadVariableOpขdense_85/MatMul/ReadVariableOpขdense_86/BiasAdd/ReadVariableOpขdense_86/MatMul/ReadVariableOpa
flatten_22/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   i
flatten_22/ReshapeReshapeinputsflatten_22/Const:output:0*
T0*
_output_shapes

:
dense_82/MatMul/ReadVariableOpReadVariableOp'dense_82_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_82/MatMulMatMulflatten_22/Reshape:output:0&dense_82/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_82/BiasAdd/ReadVariableOpReadVariableOp(dense_82_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_82/BiasAddBiasAdddense_82/MatMul:product:0'dense_82/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Y
dense_82/ReluReludense_82/BiasAdd:output:0*
T0*
_output_shapes

:
dense_83/MatMul/ReadVariableOpReadVariableOp'dense_83_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_83/MatMulMatMuldense_82/Relu:activations:0&dense_83/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_83/BiasAdd/ReadVariableOpReadVariableOp(dense_83_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_83/BiasAddBiasAdddense_83/MatMul:product:0'dense_83/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: Y
dense_83/ReluReludense_83/BiasAdd:output:0*
T0*
_output_shapes

: 
dense_84/MatMul/ReadVariableOpReadVariableOp'dense_84_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_84/MatMulMatMuldense_83/Relu:activations:0&dense_84/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_84/BiasAdd/ReadVariableOpReadVariableOp(dense_84_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_84/BiasAddBiasAdddense_84/MatMul:product:0'dense_84/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: Y
dense_84/ReluReludense_84/BiasAdd:output:0*
T0*
_output_shapes

: 
dense_85/MatMul/ReadVariableOpReadVariableOp'dense_85_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_85/MatMulMatMuldense_84/Relu:activations:0&dense_85/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_85/BiasAdd/ReadVariableOpReadVariableOp(dense_85_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_85/BiasAddBiasAdddense_85/MatMul:product:0'dense_85/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: Y
dense_85/ReluReludense_85/BiasAdd:output:0*
T0*
_output_shapes

: 
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_86/MatMulMatMuldense_85/Relu:activations:0&dense_86/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_86/BiasAdd/ReadVariableOpReadVariableOp(dense_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_86/BiasAddBiasAdddense_86/MatMul:product:0'dense_86/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_
IdentityIdentitydense_86/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp ^dense_82/BiasAdd/ReadVariableOp^dense_82/MatMul/ReadVariableOp ^dense_83/BiasAdd/ReadVariableOp^dense_83/MatMul/ReadVariableOp ^dense_84/BiasAdd/ReadVariableOp^dense_84/MatMul/ReadVariableOp ^dense_85/BiasAdd/ReadVariableOp^dense_85/MatMul/ReadVariableOp ^dense_86/BiasAdd/ReadVariableOp^dense_86/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2B
dense_82/BiasAdd/ReadVariableOpdense_82/BiasAdd/ReadVariableOp2@
dense_82/MatMul/ReadVariableOpdense_82/MatMul/ReadVariableOp2B
dense_83/BiasAdd/ReadVariableOpdense_83/BiasAdd/ReadVariableOp2@
dense_83/MatMul/ReadVariableOpdense_83/MatMul/ReadVariableOp2B
dense_84/BiasAdd/ReadVariableOpdense_84/BiasAdd/ReadVariableOp2@
dense_84/MatMul/ReadVariableOpdense_84/MatMul/ReadVariableOp2B
dense_85/BiasAdd/ReadVariableOpdense_85/BiasAdd/ReadVariableOp2@
dense_85/MatMul/ReadVariableOpdense_85/MatMul/ReadVariableOp2B
dense_86/BiasAdd/ReadVariableOpdense_86/BiasAdd/ReadVariableOp2@
dense_86/MatMul/ReadVariableOpdense_86/MatMul/ReadVariableOp:J F
"
_output_shapes
:
 
_user_specified_nameinputs
ๆ	
๖
E__inference_dense_85_layer_call_and_return_conditional_losses_1283377

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

: X
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

: w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

: 
 
_user_specified_nameinputs

๔
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1283015

inputs"
dense_82_1282989:
dense_82_1282991:"
dense_83_1282994: 
dense_83_1282996: "
dense_84_1282999:  
dense_84_1283001: "
dense_85_1283004:  
dense_85_1283006: "
dense_86_1283009: 
dense_86_1283011:
identityข dense_82/StatefulPartitionedCallข dense_83/StatefulPartitionedCallข dense_84/StatefulPartitionedCallข dense_85/StatefulPartitionedCallข dense_86/StatefulPartitionedCallด
flatten_22/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_22_layer_call_and_return_conditional_losses_1282792
 dense_82/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_82_1282989dense_82_1282991*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_82_layer_call_and_return_conditional_losses_1282805
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_1282994dense_83_1282996*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_83_layer_call_and_return_conditional_losses_1282822
 dense_84/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0dense_84_1282999dense_84_1283001*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_84_layer_call_and_return_conditional_losses_1282839
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0dense_85_1283004dense_85_1283006*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_85_layer_call_and_return_conditional_losses_1282856
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0dense_86_1283009dense_86_1283011*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_86_layer_call_and_return_conditional_losses_1282872o
IdentityIdentity)dense_86/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:๕
NoOpNoOp!^dense_82/StatefulPartitionedCall!^dense_83/StatefulPartitionedCall!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs


?
7__inference_Flatten5Layers_batch4_layer_call_fn_1283206

inputs
unknown:
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8:
identityขStatefulPartitionedCallฦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1283015f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs

๖
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1283093
input_31"
dense_82_1283067:
dense_82_1283069:"
dense_83_1283072: 
dense_83_1283074: "
dense_84_1283077:  
dense_84_1283079: "
dense_85_1283082:  
dense_85_1283084: "
dense_86_1283087: 
dense_86_1283089:
identityข dense_82/StatefulPartitionedCallข dense_83/StatefulPartitionedCallข dense_84/StatefulPartitionedCallข dense_85/StatefulPartitionedCallข dense_86/StatefulPartitionedCallถ
flatten_22/PartitionedCallPartitionedCallinput_31*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_22_layer_call_and_return_conditional_losses_1282792
 dense_82/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_82_1283067dense_82_1283069*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_82_layer_call_and_return_conditional_losses_1282805
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_1283072dense_83_1283074*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_83_layer_call_and_return_conditional_losses_1282822
 dense_84/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0dense_84_1283077dense_84_1283079*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_84_layer_call_and_return_conditional_losses_1282839
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0dense_85_1283082dense_85_1283084*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_85_layer_call_and_return_conditional_losses_1282856
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0dense_86_1283087dense_86_1283089*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_86_layer_call_and_return_conditional_losses_1282872o
IdentityIdentity)dense_86/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:๕
NoOpNoOp!^dense_82/StatefulPartitionedCall!^dense_83/StatefulPartitionedCall!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall:L H
"
_output_shapes
:
"
_user_specified_name
input_31
ๆ	
๖
E__inference_dense_83_layer_call_and_return_conditional_losses_1283337

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

: X
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

: w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
ๆ	
๖
E__inference_dense_82_layer_call_and_return_conditional_losses_1283317

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:X
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

:w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
?

*__inference_dense_82_layer_call_fn_1283306

inputs
unknown:
	unknown_0:
identityขStatefulPartitionedCallั
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_82_layer_call_and_return_conditional_losses_1282805f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs


?
7__inference_Flatten5Layers_batch4_layer_call_fn_1283181

inputs
unknown:
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8:
identityขStatefulPartitionedCallฦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1282879f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs

c
G__inference_flatten_22_layer_call_and_return_conditional_losses_1283297

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   S
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes

:O
IdentityIdentityReshape:output:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
แM
ง
 __inference__traced_save_1283530
file_prefix.
*savev2_dense_82_kernel_read_readvariableop,
(savev2_dense_82_bias_read_readvariableop.
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
1savev2_adam_dense_82_kernel_m_read_readvariableop3
/savev2_adam_dense_82_bias_m_read_readvariableop5
1savev2_adam_dense_83_kernel_m_read_readvariableop3
/savev2_adam_dense_83_bias_m_read_readvariableop5
1savev2_adam_dense_84_kernel_m_read_readvariableop3
/savev2_adam_dense_84_bias_m_read_readvariableop5
1savev2_adam_dense_85_kernel_m_read_readvariableop3
/savev2_adam_dense_85_bias_m_read_readvariableop5
1savev2_adam_dense_86_kernel_m_read_readvariableop3
/savev2_adam_dense_86_bias_m_read_readvariableop5
1savev2_adam_dense_82_kernel_v_read_readvariableop3
/savev2_adam_dense_82_bias_v_read_readvariableop5
1savev2_adam_dense_83_kernel_v_read_readvariableop3
/savev2_adam_dense_83_bias_v_read_readvariableop5
1savev2_adam_dense_84_kernel_v_read_readvariableop3
/savev2_adam_dense_84_bias_v_read_readvariableop5
1savev2_adam_dense_85_kernel_v_read_readvariableop3
/savev2_adam_dense_85_bias_v_read_readvariableop5
1savev2_adam_dense_86_kernel_v_read_readvariableop3
/savev2_adam_dense_86_bias_v_read_readvariableop
savev2_const

identity_1ขMergeV2Checkpointsw
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
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*ฆ
valueB&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHน
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ๛
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_82_kernel_read_readvariableop(savev2_dense_82_bias_read_readvariableop*savev2_dense_83_kernel_read_readvariableop(savev2_dense_83_bias_read_readvariableop*savev2_dense_84_kernel_read_readvariableop(savev2_dense_84_bias_read_readvariableop*savev2_dense_85_kernel_read_readvariableop(savev2_dense_85_bias_read_readvariableop*savev2_dense_86_kernel_read_readvariableop(savev2_dense_86_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_82_kernel_m_read_readvariableop/savev2_adam_dense_82_bias_m_read_readvariableop1savev2_adam_dense_83_kernel_m_read_readvariableop/savev2_adam_dense_83_bias_m_read_readvariableop1savev2_adam_dense_84_kernel_m_read_readvariableop/savev2_adam_dense_84_bias_m_read_readvariableop1savev2_adam_dense_85_kernel_m_read_readvariableop/savev2_adam_dense_85_bias_m_read_readvariableop1savev2_adam_dense_86_kernel_m_read_readvariableop/savev2_adam_dense_86_bias_m_read_readvariableop1savev2_adam_dense_82_kernel_v_read_readvariableop/savev2_adam_dense_82_bias_v_read_readvariableop1savev2_adam_dense_83_kernel_v_read_readvariableop/savev2_adam_dense_83_bias_v_read_readvariableop1savev2_adam_dense_84_kernel_v_read_readvariableop/savev2_adam_dense_84_bias_v_read_readvariableop1savev2_adam_dense_85_kernel_v_read_readvariableop/savev2_adam_dense_85_bias_v_read_readvariableop1savev2_adam_dense_86_kernel_v_read_readvariableop/savev2_adam_dense_86_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
: ::: : :  : :  : : :: : : : : : : ::: : :  : :  : : :::: : :  : :  : : :: 2(
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

:: 

_output_shapes
::$ 

_output_shapes

: : 
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

:: 

_output_shapes
::$ 

_output_shapes

: : 
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
ำ?

"__inference__wrapped_model_1282779
input_31O
=flatten5layers_batch4_dense_82_matmul_readvariableop_resource:L
>flatten5layers_batch4_dense_82_biasadd_readvariableop_resource:O
=flatten5layers_batch4_dense_83_matmul_readvariableop_resource: L
>flatten5layers_batch4_dense_83_biasadd_readvariableop_resource: O
=flatten5layers_batch4_dense_84_matmul_readvariableop_resource:  L
>flatten5layers_batch4_dense_84_biasadd_readvariableop_resource: O
=flatten5layers_batch4_dense_85_matmul_readvariableop_resource:  L
>flatten5layers_batch4_dense_85_biasadd_readvariableop_resource: O
=flatten5layers_batch4_dense_86_matmul_readvariableop_resource: L
>flatten5layers_batch4_dense_86_biasadd_readvariableop_resource:
identityข5Flatten5Layers_batch4/dense_82/BiasAdd/ReadVariableOpข4Flatten5Layers_batch4/dense_82/MatMul/ReadVariableOpข5Flatten5Layers_batch4/dense_83/BiasAdd/ReadVariableOpข4Flatten5Layers_batch4/dense_83/MatMul/ReadVariableOpข5Flatten5Layers_batch4/dense_84/BiasAdd/ReadVariableOpข4Flatten5Layers_batch4/dense_84/MatMul/ReadVariableOpข5Flatten5Layers_batch4/dense_85/BiasAdd/ReadVariableOpข4Flatten5Layers_batch4/dense_85/MatMul/ReadVariableOpข5Flatten5Layers_batch4/dense_86/BiasAdd/ReadVariableOpข4Flatten5Layers_batch4/dense_86/MatMul/ReadVariableOpw
&Flatten5Layers_batch4/flatten_22/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   
(Flatten5Layers_batch4/flatten_22/ReshapeReshapeinput_31/Flatten5Layers_batch4/flatten_22/Const:output:0*
T0*
_output_shapes

:ฒ
4Flatten5Layers_batch4/dense_82/MatMul/ReadVariableOpReadVariableOp=flatten5layers_batch4_dense_82_matmul_readvariableop_resource*
_output_shapes

:*
dtype0ษ
%Flatten5Layers_batch4/dense_82/MatMulMatMul1Flatten5Layers_batch4/flatten_22/Reshape:output:0<Flatten5Layers_batch4/dense_82/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:ฐ
5Flatten5Layers_batch4/dense_82/BiasAdd/ReadVariableOpReadVariableOp>flatten5layers_batch4_dense_82_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ส
&Flatten5Layers_batch4/dense_82/BiasAddBiasAdd/Flatten5Layers_batch4/dense_82/MatMul:product:0=Flatten5Layers_batch4/dense_82/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
#Flatten5Layers_batch4/dense_82/ReluRelu/Flatten5Layers_batch4/dense_82/BiasAdd:output:0*
T0*
_output_shapes

:ฒ
4Flatten5Layers_batch4/dense_83/MatMul/ReadVariableOpReadVariableOp=flatten5layers_batch4_dense_83_matmul_readvariableop_resource*
_output_shapes

: *
dtype0ษ
%Flatten5Layers_batch4/dense_83/MatMulMatMul1Flatten5Layers_batch4/dense_82/Relu:activations:0<Flatten5Layers_batch4/dense_83/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ฐ
5Flatten5Layers_batch4/dense_83/BiasAdd/ReadVariableOpReadVariableOp>flatten5layers_batch4_dense_83_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ส
&Flatten5Layers_batch4/dense_83/BiasAddBiasAdd/Flatten5Layers_batch4/dense_83/MatMul:product:0=Flatten5Layers_batch4/dense_83/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
#Flatten5Layers_batch4/dense_83/ReluRelu/Flatten5Layers_batch4/dense_83/BiasAdd:output:0*
T0*
_output_shapes

: ฒ
4Flatten5Layers_batch4/dense_84/MatMul/ReadVariableOpReadVariableOp=flatten5layers_batch4_dense_84_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0ษ
%Flatten5Layers_batch4/dense_84/MatMulMatMul1Flatten5Layers_batch4/dense_83/Relu:activations:0<Flatten5Layers_batch4/dense_84/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ฐ
5Flatten5Layers_batch4/dense_84/BiasAdd/ReadVariableOpReadVariableOp>flatten5layers_batch4_dense_84_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ส
&Flatten5Layers_batch4/dense_84/BiasAddBiasAdd/Flatten5Layers_batch4/dense_84/MatMul:product:0=Flatten5Layers_batch4/dense_84/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
#Flatten5Layers_batch4/dense_84/ReluRelu/Flatten5Layers_batch4/dense_84/BiasAdd:output:0*
T0*
_output_shapes

: ฒ
4Flatten5Layers_batch4/dense_85/MatMul/ReadVariableOpReadVariableOp=flatten5layers_batch4_dense_85_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0ษ
%Flatten5Layers_batch4/dense_85/MatMulMatMul1Flatten5Layers_batch4/dense_84/Relu:activations:0<Flatten5Layers_batch4/dense_85/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ฐ
5Flatten5Layers_batch4/dense_85/BiasAdd/ReadVariableOpReadVariableOp>flatten5layers_batch4_dense_85_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ส
&Flatten5Layers_batch4/dense_85/BiasAddBiasAdd/Flatten5Layers_batch4/dense_85/MatMul:product:0=Flatten5Layers_batch4/dense_85/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
#Flatten5Layers_batch4/dense_85/ReluRelu/Flatten5Layers_batch4/dense_85/BiasAdd:output:0*
T0*
_output_shapes

: ฒ
4Flatten5Layers_batch4/dense_86/MatMul/ReadVariableOpReadVariableOp=flatten5layers_batch4_dense_86_matmul_readvariableop_resource*
_output_shapes

: *
dtype0ษ
%Flatten5Layers_batch4/dense_86/MatMulMatMul1Flatten5Layers_batch4/dense_85/Relu:activations:0<Flatten5Layers_batch4/dense_86/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:ฐ
5Flatten5Layers_batch4/dense_86/BiasAdd/ReadVariableOpReadVariableOp>flatten5layers_batch4_dense_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ส
&Flatten5Layers_batch4/dense_86/BiasAddBiasAdd/Flatten5Layers_batch4/dense_86/MatMul:product:0=Flatten5Layers_batch4/dense_86/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:u
IdentityIdentity/Flatten5Layers_batch4/dense_86/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:๑
NoOpNoOp6^Flatten5Layers_batch4/dense_82/BiasAdd/ReadVariableOp5^Flatten5Layers_batch4/dense_82/MatMul/ReadVariableOp6^Flatten5Layers_batch4/dense_83/BiasAdd/ReadVariableOp5^Flatten5Layers_batch4/dense_83/MatMul/ReadVariableOp6^Flatten5Layers_batch4/dense_84/BiasAdd/ReadVariableOp5^Flatten5Layers_batch4/dense_84/MatMul/ReadVariableOp6^Flatten5Layers_batch4/dense_85/BiasAdd/ReadVariableOp5^Flatten5Layers_batch4/dense_85/MatMul/ReadVariableOp6^Flatten5Layers_batch4/dense_86/BiasAdd/ReadVariableOp5^Flatten5Layers_batch4/dense_86/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2n
5Flatten5Layers_batch4/dense_82/BiasAdd/ReadVariableOp5Flatten5Layers_batch4/dense_82/BiasAdd/ReadVariableOp2l
4Flatten5Layers_batch4/dense_82/MatMul/ReadVariableOp4Flatten5Layers_batch4/dense_82/MatMul/ReadVariableOp2n
5Flatten5Layers_batch4/dense_83/BiasAdd/ReadVariableOp5Flatten5Layers_batch4/dense_83/BiasAdd/ReadVariableOp2l
4Flatten5Layers_batch4/dense_83/MatMul/ReadVariableOp4Flatten5Layers_batch4/dense_83/MatMul/ReadVariableOp2n
5Flatten5Layers_batch4/dense_84/BiasAdd/ReadVariableOp5Flatten5Layers_batch4/dense_84/BiasAdd/ReadVariableOp2l
4Flatten5Layers_batch4/dense_84/MatMul/ReadVariableOp4Flatten5Layers_batch4/dense_84/MatMul/ReadVariableOp2n
5Flatten5Layers_batch4/dense_85/BiasAdd/ReadVariableOp5Flatten5Layers_batch4/dense_85/BiasAdd/ReadVariableOp2l
4Flatten5Layers_batch4/dense_85/MatMul/ReadVariableOp4Flatten5Layers_batch4/dense_85/MatMul/ReadVariableOp2n
5Flatten5Layers_batch4/dense_86/BiasAdd/ReadVariableOp5Flatten5Layers_batch4/dense_86/BiasAdd/ReadVariableOp2l
4Flatten5Layers_batch4/dense_86/MatMul/ReadVariableOp4Flatten5Layers_batch4/dense_86/MatMul/ReadVariableOp:L H
"
_output_shapes
:
"
_user_specified_name
input_31
ึ	
์
%__inference_signature_wrapper_1283156
input_31
unknown:
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8:
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_31unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*,
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
"__inference__wrapped_model_1282779f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
"
_output_shapes
:
"
_user_specified_name
input_31
ๆ	
๖
E__inference_dense_85_layer_call_and_return_conditional_losses_1282856

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

: X
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

: w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

: 
 
_user_specified_nameinputs
ๆ	
๖
E__inference_dense_83_layer_call_and_return_conditional_losses_1282822

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

: X
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

: w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
?

*__inference_dense_83_layer_call_fn_1283326

inputs
unknown: 
	unknown_0: 
identityขStatefulPartitionedCallั
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_83_layer_call_and_return_conditional_losses_1282822f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs

c
G__inference_flatten_22_layer_call_and_return_conditional_losses_1282792

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   S
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes

:O
IdentityIdentityReshape:output:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
?

*__inference_dense_84_layer_call_fn_1283346

inputs
unknown:  
	unknown_0: 
identityขStatefulPartitionedCallั
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_84_layer_call_and_return_conditional_losses_1282839f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

: 
 
_user_specified_nameinputs
ุ,
?
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1283286

inputs9
'dense_82_matmul_readvariableop_resource:6
(dense_82_biasadd_readvariableop_resource:9
'dense_83_matmul_readvariableop_resource: 6
(dense_83_biasadd_readvariableop_resource: 9
'dense_84_matmul_readvariableop_resource:  6
(dense_84_biasadd_readvariableop_resource: 9
'dense_85_matmul_readvariableop_resource:  6
(dense_85_biasadd_readvariableop_resource: 9
'dense_86_matmul_readvariableop_resource: 6
(dense_86_biasadd_readvariableop_resource:
identityขdense_82/BiasAdd/ReadVariableOpขdense_82/MatMul/ReadVariableOpขdense_83/BiasAdd/ReadVariableOpขdense_83/MatMul/ReadVariableOpขdense_84/BiasAdd/ReadVariableOpขdense_84/MatMul/ReadVariableOpขdense_85/BiasAdd/ReadVariableOpขdense_85/MatMul/ReadVariableOpขdense_86/BiasAdd/ReadVariableOpขdense_86/MatMul/ReadVariableOpa
flatten_22/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   i
flatten_22/ReshapeReshapeinputsflatten_22/Const:output:0*
T0*
_output_shapes

:
dense_82/MatMul/ReadVariableOpReadVariableOp'dense_82_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_82/MatMulMatMulflatten_22/Reshape:output:0&dense_82/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_82/BiasAdd/ReadVariableOpReadVariableOp(dense_82_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_82/BiasAddBiasAdddense_82/MatMul:product:0'dense_82/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Y
dense_82/ReluReludense_82/BiasAdd:output:0*
T0*
_output_shapes

:
dense_83/MatMul/ReadVariableOpReadVariableOp'dense_83_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_83/MatMulMatMuldense_82/Relu:activations:0&dense_83/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_83/BiasAdd/ReadVariableOpReadVariableOp(dense_83_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_83/BiasAddBiasAdddense_83/MatMul:product:0'dense_83/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: Y
dense_83/ReluReludense_83/BiasAdd:output:0*
T0*
_output_shapes

: 
dense_84/MatMul/ReadVariableOpReadVariableOp'dense_84_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_84/MatMulMatMuldense_83/Relu:activations:0&dense_84/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_84/BiasAdd/ReadVariableOpReadVariableOp(dense_84_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_84/BiasAddBiasAdddense_84/MatMul:product:0'dense_84/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: Y
dense_84/ReluReludense_84/BiasAdd:output:0*
T0*
_output_shapes

: 
dense_85/MatMul/ReadVariableOpReadVariableOp'dense_85_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_85/MatMulMatMuldense_84/Relu:activations:0&dense_85/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_85/BiasAdd/ReadVariableOpReadVariableOp(dense_85_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_85/BiasAddBiasAdddense_85/MatMul:product:0'dense_85/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: Y
dense_85/ReluReludense_85/BiasAdd:output:0*
T0*
_output_shapes

: 
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_86/MatMulMatMuldense_85/Relu:activations:0&dense_86/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_86/BiasAdd/ReadVariableOpReadVariableOp(dense_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_86/BiasAddBiasAdddense_86/MatMul:product:0'dense_86/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_
IdentityIdentitydense_86/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp ^dense_82/BiasAdd/ReadVariableOp^dense_82/MatMul/ReadVariableOp ^dense_83/BiasAdd/ReadVariableOp^dense_83/MatMul/ReadVariableOp ^dense_84/BiasAdd/ReadVariableOp^dense_84/MatMul/ReadVariableOp ^dense_85/BiasAdd/ReadVariableOp^dense_85/MatMul/ReadVariableOp ^dense_86/BiasAdd/ReadVariableOp^dense_86/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2B
dense_82/BiasAdd/ReadVariableOpdense_82/BiasAdd/ReadVariableOp2@
dense_82/MatMul/ReadVariableOpdense_82/MatMul/ReadVariableOp2B
dense_83/BiasAdd/ReadVariableOpdense_83/BiasAdd/ReadVariableOp2@
dense_83/MatMul/ReadVariableOpdense_83/MatMul/ReadVariableOp2B
dense_84/BiasAdd/ReadVariableOpdense_84/BiasAdd/ReadVariableOp2@
dense_84/MatMul/ReadVariableOpdense_84/MatMul/ReadVariableOp2B
dense_85/BiasAdd/ReadVariableOpdense_85/BiasAdd/ReadVariableOp2@
dense_85/MatMul/ReadVariableOpdense_85/MatMul/ReadVariableOp2B
dense_86/BiasAdd/ReadVariableOpdense_86/BiasAdd/ReadVariableOp2@
dense_86/MatMul/ReadVariableOpdense_86/MatMul/ReadVariableOp:J F
"
_output_shapes
:
 
_user_specified_nameinputs
ๆ	
๖
E__inference_dense_84_layer_call_and_return_conditional_losses_1283357

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

: X
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

: w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

: 
 
_user_specified_nameinputs

H
,__inference_flatten_22_layer_call_fn_1283291

inputs
identityฉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_22_layer_call_and_return_conditional_losses_1282792W
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs

๖
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1283123
input_31"
dense_82_1283097:
dense_82_1283099:"
dense_83_1283102: 
dense_83_1283104: "
dense_84_1283107:  
dense_84_1283109: "
dense_85_1283112:  
dense_85_1283114: "
dense_86_1283117: 
dense_86_1283119:
identityข dense_82/StatefulPartitionedCallข dense_83/StatefulPartitionedCallข dense_84/StatefulPartitionedCallข dense_85/StatefulPartitionedCallข dense_86/StatefulPartitionedCallถ
flatten_22/PartitionedCallPartitionedCallinput_31*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_22_layer_call_and_return_conditional_losses_1282792
 dense_82/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_82_1283097dense_82_1283099*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_82_layer_call_and_return_conditional_losses_1282805
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_1283102dense_83_1283104*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_83_layer_call_and_return_conditional_losses_1282822
 dense_84/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0dense_84_1283107dense_84_1283109*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_84_layer_call_and_return_conditional_losses_1282839
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0dense_85_1283112dense_85_1283114*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_85_layer_call_and_return_conditional_losses_1282856
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0dense_86_1283117dense_86_1283119*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_86_layer_call_and_return_conditional_losses_1282872o
IdentityIdentity)dense_86/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:๕
NoOpNoOp!^dense_82/StatefulPartitionedCall!^dense_83/StatefulPartitionedCall!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall:L H
"
_output_shapes
:
"
_user_specified_name
input_31
ๆ	
๖
E__inference_dense_84_layer_call_and_return_conditional_losses_1282839

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

: X
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

: w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

: 
 
_user_specified_nameinputs"ต	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
8
input_31,
serving_default_input_31:03
dense_86'
StatefulPartitionedCall:0tensorflow/serving/predict:ก
ถ
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
ฅ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
ป
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ป
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
ป
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias"
_tf_keras_layer
ป
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias"
_tf_keras_layer
ป
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
ส
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

Ctrace_0
Dtrace_1
Etrace_2
Ftrace_32ฆ
7__inference_Flatten5Layers_batch4_layer_call_fn_1282902
7__inference_Flatten5Layers_batch4_layer_call_fn_1283181
7__inference_Flatten5Layers_batch4_layer_call_fn_1283206
7__inference_Flatten5Layers_batch4_layer_call_fn_1283063ฟ
ถฒฒ
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
annotationsช *
 zCtrace_0zDtrace_1zEtrace_2zFtrace_3
?
Gtrace_0
Htrace_1
Itrace_2
Jtrace_32
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1283246
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1283286
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1283093
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1283123ฟ
ถฒฒ
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
annotationsช *
 zGtrace_0zHtrace_1zItrace_2zJtrace_3
ฮBห
"__inference__wrapped_model_1282779input_31"
ฒ
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
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
ญ
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
๐
Vtrace_02ำ
,__inference_flatten_22_layer_call_fn_1283291ข
ฒ
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
annotationsช *
 zVtrace_0

Wtrace_02๎
G__inference_flatten_22_layer_call_and_return_conditional_losses_1283297ข
ฒ
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
annotationsช *
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
ญ
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
๎
]trace_02ั
*__inference_dense_82_layer_call_fn_1283306ข
ฒ
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
annotationsช *
 z]trace_0

^trace_02์
E__inference_dense_82_layer_call_and_return_conditional_losses_1283317ข
ฒ
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
annotationsช *
 z^trace_0
!:2dense_82/kernel
:2dense_82/bias
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
ญ
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
๎
dtrace_02ั
*__inference_dense_83_layer_call_fn_1283326ข
ฒ
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
annotationsช *
 zdtrace_0

etrace_02์
E__inference_dense_83_layer_call_and_return_conditional_losses_1283337ข
ฒ
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
annotationsช *
 zetrace_0
!: 2dense_83/kernel
: 2dense_83/bias
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
ญ
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
๎
ktrace_02ั
*__inference_dense_84_layer_call_fn_1283346ข
ฒ
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
annotationsช *
 zktrace_0

ltrace_02์
E__inference_dense_84_layer_call_and_return_conditional_losses_1283357ข
ฒ
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
annotationsช *
 zltrace_0
!:  2dense_84/kernel
: 2dense_84/bias
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
ญ
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
๎
rtrace_02ั
*__inference_dense_85_layer_call_fn_1283366ข
ฒ
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
annotationsช *
 zrtrace_0

strace_02์
E__inference_dense_85_layer_call_and_return_conditional_losses_1283377ข
ฒ
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
annotationsช *
 zstrace_0
!:  2dense_85/kernel
: 2dense_85/bias
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
ญ
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
๎
ytrace_02ั
*__inference_dense_86_layer_call_fn_1283386ข
ฒ
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
annotationsช *
 zytrace_0

ztrace_02์
E__inference_dense_86_layer_call_and_return_conditional_losses_1283396ข
ฒ
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
annotationsช *
 zztrace_0
!: 2dense_86/kernel
:2dense_86/bias
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
B
7__inference_Flatten5Layers_batch4_layer_call_fn_1282902input_31"ฟ
ถฒฒ
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
annotationsช *
 
B
7__inference_Flatten5Layers_batch4_layer_call_fn_1283181inputs"ฟ
ถฒฒ
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
annotationsช *
 
B
7__inference_Flatten5Layers_batch4_layer_call_fn_1283206inputs"ฟ
ถฒฒ
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
annotationsช *
 
B
7__inference_Flatten5Layers_batch4_layer_call_fn_1283063input_31"ฟ
ถฒฒ
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
annotationsช *
 
ฃB?
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1283246inputs"ฟ
ถฒฒ
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
annotationsช *
 
ฃB?
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1283286inputs"ฟ
ถฒฒ
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
annotationsช *
 
ฅBข
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1283093input_31"ฟ
ถฒฒ
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
annotationsช *
 
ฅBข
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1283123input_31"ฟ
ถฒฒ
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
annotationsช *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
อBส
%__inference_signature_wrapper_1283156input_31"
ฒ
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
annotationsช *
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
เB?
,__inference_flatten_22_layer_call_fn_1283291inputs"ข
ฒ
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
annotationsช *
 
๛B๘
G__inference_flatten_22_layer_call_and_return_conditional_losses_1283297inputs"ข
ฒ
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
annotationsช *
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
?B?
*__inference_dense_82_layer_call_fn_1283306inputs"ข
ฒ
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
annotationsช *
 
๙B๖
E__inference_dense_82_layer_call_and_return_conditional_losses_1283317inputs"ข
ฒ
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
annotationsช *
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
?B?
*__inference_dense_83_layer_call_fn_1283326inputs"ข
ฒ
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
annotationsช *
 
๙B๖
E__inference_dense_83_layer_call_and_return_conditional_losses_1283337inputs"ข
ฒ
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
annotationsช *
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
?B?
*__inference_dense_84_layer_call_fn_1283346inputs"ข
ฒ
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
annotationsช *
 
๙B๖
E__inference_dense_84_layer_call_and_return_conditional_losses_1283357inputs"ข
ฒ
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
annotationsช *
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
?B?
*__inference_dense_85_layer_call_fn_1283366inputs"ข
ฒ
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
annotationsช *
 
๙B๖
E__inference_dense_85_layer_call_and_return_conditional_losses_1283377inputs"ข
ฒ
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
annotationsช *
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
?B?
*__inference_dense_86_layer_call_fn_1283386inputs"ข
ฒ
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
annotationsช *
 
๙B๖
E__inference_dense_86_layer_call_and_return_conditional_losses_1283396inputs"ข
ฒ
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
annotationsช *
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
&:$2Adam/dense_82/kernel/m
 :2Adam/dense_82/bias/m
&:$ 2Adam/dense_83/kernel/m
 : 2Adam/dense_83/bias/m
&:$  2Adam/dense_84/kernel/m
 : 2Adam/dense_84/bias/m
&:$  2Adam/dense_85/kernel/m
 : 2Adam/dense_85/bias/m
&:$ 2Adam/dense_86/kernel/m
 :2Adam/dense_86/bias/m
&:$2Adam/dense_82/kernel/v
 :2Adam/dense_82/bias/v
&:$ 2Adam/dense_83/kernel/v
 : 2Adam/dense_83/bias/v
&:$  2Adam/dense_84/kernel/v
 : 2Adam/dense_84/bias/v
&:$  2Adam/dense_85/kernel/v
 : 2Adam/dense_85/bias/v
&:$ 2Adam/dense_86/kernel/v
 :2Adam/dense_86/bias/vถ
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1283093`
$%,-45<=4ข1
*ข'

input_31
p 

 
ช "ข

0
 ถ
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1283123`
$%,-45<=4ข1
*ข'

input_31
p

 
ช "ข

0
 ด
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1283246^
$%,-45<=2ข/
(ข%

inputs
p 

 
ช "ข

0
 ด
R__inference_Flatten5Layers_batch4_layer_call_and_return_conditional_losses_1283286^
$%,-45<=2ข/
(ข%

inputs
p

 
ช "ข

0
 
7__inference_Flatten5Layers_batch4_layer_call_fn_1282902S
$%,-45<=4ข1
*ข'

input_31
p 

 
ช "
7__inference_Flatten5Layers_batch4_layer_call_fn_1283063S
$%,-45<=4ข1
*ข'

input_31
p

 
ช "
7__inference_Flatten5Layers_batch4_layer_call_fn_1283181Q
$%,-45<=2ข/
(ข%

inputs
p 

 
ช "
7__inference_Flatten5Layers_batch4_layer_call_fn_1283206Q
$%,-45<=2ข/
(ข%

inputs
p

 
ช "
"__inference__wrapped_model_1282779f
$%,-45<=,ข)
"ข

input_31
ช "*ช'
%
dense_86
dense_86
E__inference_dense_82_layer_call_and_return_conditional_losses_1283317J&ข#
ข

inputs
ช "ข

0
 k
*__inference_dense_82_layer_call_fn_1283306=&ข#
ข

inputs
ช "
E__inference_dense_83_layer_call_and_return_conditional_losses_1283337J$%&ข#
ข

inputs
ช "ข

0 
 k
*__inference_dense_83_layer_call_fn_1283326=$%&ข#
ข

inputs
ช " 
E__inference_dense_84_layer_call_and_return_conditional_losses_1283357J,-&ข#
ข

inputs 
ช "ข

0 
 k
*__inference_dense_84_layer_call_fn_1283346=,-&ข#
ข

inputs 
ช " 
E__inference_dense_85_layer_call_and_return_conditional_losses_1283377J45&ข#
ข

inputs 
ช "ข

0 
 k
*__inference_dense_85_layer_call_fn_1283366=45&ข#
ข

inputs 
ช " 
E__inference_dense_86_layer_call_and_return_conditional_losses_1283396J<=&ข#
ข

inputs 
ช "ข

0
 k
*__inference_dense_86_layer_call_fn_1283386=<=&ข#
ข

inputs 
ช "
G__inference_flatten_22_layer_call_and_return_conditional_losses_1283297J*ข'
 ข

inputs
ช "ข

0
 m
,__inference_flatten_22_layer_call_fn_1283291=*ข'
 ข

inputs
ช "
%__inference_signature_wrapper_1283156r
$%,-45<=8ข5
ข 
.ช+
)
input_31
input_31"*ช'
%
dense_86
dense_86