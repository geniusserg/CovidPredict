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
Adam/dense_76/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_76/bias/v
y
(Adam/dense_76/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_76/bias/v*
_output_shapes
:*
dtype0

Adam/dense_76/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_76/kernel/v

*Adam/dense_76/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_76/kernel/v*
_output_shapes

: *
dtype0

Adam/dense_75/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_75/bias/v
y
(Adam/dense_75/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_75/bias/v*
_output_shapes
: *
dtype0

Adam/dense_75/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_75/kernel/v

*Adam/dense_75/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_75/kernel/v*
_output_shapes

:  *
dtype0

Adam/dense_74/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_74/bias/v
y
(Adam/dense_74/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_74/bias/v*
_output_shapes
: *
dtype0

Adam/dense_74/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_74/kernel/v

*Adam/dense_74/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_74/kernel/v*
_output_shapes

:  *
dtype0

Adam/dense_73/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_73/bias/v
y
(Adam/dense_73/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_73/bias/v*
_output_shapes
: *
dtype0

Adam/dense_73/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_73/kernel/v

*Adam/dense_73/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_73/kernel/v*
_output_shapes

: *
dtype0

Adam/dense_72/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_72/bias/v
y
(Adam/dense_72/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_72/bias/v*
_output_shapes
:*
dtype0

Adam/dense_72/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_72/kernel/v

*Adam/dense_72/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_72/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_76/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_76/bias/m
y
(Adam/dense_76/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_76/bias/m*
_output_shapes
:*
dtype0

Adam/dense_76/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_76/kernel/m

*Adam/dense_76/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_76/kernel/m*
_output_shapes

: *
dtype0

Adam/dense_75/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_75/bias/m
y
(Adam/dense_75/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_75/bias/m*
_output_shapes
: *
dtype0

Adam/dense_75/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_75/kernel/m

*Adam/dense_75/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_75/kernel/m*
_output_shapes

:  *
dtype0

Adam/dense_74/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_74/bias/m
y
(Adam/dense_74/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_74/bias/m*
_output_shapes
: *
dtype0

Adam/dense_74/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_74/kernel/m

*Adam/dense_74/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_74/kernel/m*
_output_shapes

:  *
dtype0

Adam/dense_73/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_73/bias/m
y
(Adam/dense_73/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_73/bias/m*
_output_shapes
: *
dtype0

Adam/dense_73/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_73/kernel/m

*Adam/dense_73/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_73/kernel/m*
_output_shapes

: *
dtype0

Adam/dense_72/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_72/bias/m
y
(Adam/dense_72/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_72/bias/m*
_output_shapes
:*
dtype0

Adam/dense_72/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_72/kernel/m

*Adam/dense_72/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_72/kernel/m*
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
dense_76/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_76/bias
k
!dense_76/bias/Read/ReadVariableOpReadVariableOpdense_76/bias*
_output_shapes
:*
dtype0
z
dense_76/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_76/kernel
s
#dense_76/kernel/Read/ReadVariableOpReadVariableOpdense_76/kernel*
_output_shapes

: *
dtype0
r
dense_75/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_75/bias
k
!dense_75/bias/Read/ReadVariableOpReadVariableOpdense_75/bias*
_output_shapes
: *
dtype0
z
dense_75/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_75/kernel
s
#dense_75/kernel/Read/ReadVariableOpReadVariableOpdense_75/kernel*
_output_shapes

:  *
dtype0
r
dense_74/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_74/bias
k
!dense_74/bias/Read/ReadVariableOpReadVariableOpdense_74/bias*
_output_shapes
: *
dtype0
z
dense_74/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_74/kernel
s
#dense_74/kernel/Read/ReadVariableOpReadVariableOpdense_74/kernel*
_output_shapes

:  *
dtype0
r
dense_73/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_73/bias
k
!dense_73/bias/Read/ReadVariableOpReadVariableOpdense_73/bias*
_output_shapes
: *
dtype0
z
dense_73/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_73/kernel
s
#dense_73/kernel/Read/ReadVariableOpReadVariableOpdense_73/kernel*
_output_shapes

: *
dtype0
r
dense_72/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_72/bias
k
!dense_72/bias/Read/ReadVariableOpReadVariableOpdense_72/bias*
_output_shapes
:*
dtype0
z
dense_72/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_72/kernel
s
#dense_72/kernel/Read/ReadVariableOpReadVariableOpdense_72/kernel*
_output_shapes

:*
dtype0
q
serving_default_input_29Placeholder*"
_output_shapes
:*
dtype0*
shape:
฿
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_29dense_72/kerneldense_72/biasdense_73/kerneldense_73/biasdense_74/kerneldense_74/biasdense_75/kerneldense_75/biasdense_76/kerneldense_76/bias*
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
%__inference_signature_wrapper_1021460

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
VARIABLE_VALUEdense_72/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_72/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_73/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_73/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_74/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_74/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_75/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_75/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_76/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_76/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_72/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_72/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_73/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_73/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_74/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_74/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_75/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_75/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_76/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_76/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_72/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_72/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_73/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_73/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_74/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_74/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_75/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_75/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_76/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_76/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ไ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_72/kernel/Read/ReadVariableOp!dense_72/bias/Read/ReadVariableOp#dense_73/kernel/Read/ReadVariableOp!dense_73/bias/Read/ReadVariableOp#dense_74/kernel/Read/ReadVariableOp!dense_74/bias/Read/ReadVariableOp#dense_75/kernel/Read/ReadVariableOp!dense_75/bias/Read/ReadVariableOp#dense_76/kernel/Read/ReadVariableOp!dense_76/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_72/kernel/m/Read/ReadVariableOp(Adam/dense_72/bias/m/Read/ReadVariableOp*Adam/dense_73/kernel/m/Read/ReadVariableOp(Adam/dense_73/bias/m/Read/ReadVariableOp*Adam/dense_74/kernel/m/Read/ReadVariableOp(Adam/dense_74/bias/m/Read/ReadVariableOp*Adam/dense_75/kernel/m/Read/ReadVariableOp(Adam/dense_75/bias/m/Read/ReadVariableOp*Adam/dense_76/kernel/m/Read/ReadVariableOp(Adam/dense_76/bias/m/Read/ReadVariableOp*Adam/dense_72/kernel/v/Read/ReadVariableOp(Adam/dense_72/bias/v/Read/ReadVariableOp*Adam/dense_73/kernel/v/Read/ReadVariableOp(Adam/dense_73/bias/v/Read/ReadVariableOp*Adam/dense_74/kernel/v/Read/ReadVariableOp(Adam/dense_74/bias/v/Read/ReadVariableOp*Adam/dense_75/kernel/v/Read/ReadVariableOp(Adam/dense_75/bias/v/Read/ReadVariableOp*Adam/dense_76/kernel/v/Read/ReadVariableOp(Adam/dense_76/bias/v/Read/ReadVariableOpConst*2
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
 __inference__traced_save_1021834
๛
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_72/kerneldense_72/biasdense_73/kerneldense_73/biasdense_74/kerneldense_74/biasdense_75/kerneldense_75/biasdense_76/kerneldense_76/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_72/kernel/mAdam/dense_72/bias/mAdam/dense_73/kernel/mAdam/dense_73/bias/mAdam/dense_74/kernel/mAdam/dense_74/bias/mAdam/dense_75/kernel/mAdam/dense_75/bias/mAdam/dense_76/kernel/mAdam/dense_76/bias/mAdam/dense_72/kernel/vAdam/dense_72/bias/vAdam/dense_73/kernel/vAdam/dense_73/bias/vAdam/dense_74/kernel/vAdam/dense_74/bias/vAdam/dense_75/kernel/vAdam/dense_75/bias/vAdam/dense_76/kernel/vAdam/dense_76/bias/v*1
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
#__inference__traced_restore_1021955ชน
ๆ	
๖
E__inference_dense_74_layer_call_and_return_conditional_losses_1021661

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

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

: X
IdentityIdentityRelu:activations:0^NoOp*
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
ุ,
?
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021590

inputs9
'dense_72_matmul_readvariableop_resource:6
(dense_72_biasadd_readvariableop_resource:9
'dense_73_matmul_readvariableop_resource: 6
(dense_73_biasadd_readvariableop_resource: 9
'dense_74_matmul_readvariableop_resource:  6
(dense_74_biasadd_readvariableop_resource: 9
'dense_75_matmul_readvariableop_resource:  6
(dense_75_biasadd_readvariableop_resource: 9
'dense_76_matmul_readvariableop_resource: 6
(dense_76_biasadd_readvariableop_resource:
identityขdense_72/BiasAdd/ReadVariableOpขdense_72/MatMul/ReadVariableOpขdense_73/BiasAdd/ReadVariableOpขdense_73/MatMul/ReadVariableOpขdense_74/BiasAdd/ReadVariableOpขdense_74/MatMul/ReadVariableOpขdense_75/BiasAdd/ReadVariableOpขdense_75/MatMul/ReadVariableOpขdense_76/BiasAdd/ReadVariableOpขdense_76/MatMul/ReadVariableOpa
flatten_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   i
flatten_20/ReshapeReshapeinputsflatten_20/Const:output:0*
T0*
_output_shapes

:
dense_72/MatMul/ReadVariableOpReadVariableOp'dense_72_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_72/MatMulMatMulflatten_20/Reshape:output:0&dense_72/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_72/BiasAdd/ReadVariableOpReadVariableOp(dense_72_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_72/BiasAddBiasAdddense_72/MatMul:product:0'dense_72/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Y
dense_72/ReluReludense_72/BiasAdd:output:0*
T0*
_output_shapes

:
dense_73/MatMul/ReadVariableOpReadVariableOp'dense_73_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_73/MatMulMatMuldense_72/Relu:activations:0&dense_73/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_73/BiasAdd/ReadVariableOpReadVariableOp(dense_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_73/BiasAddBiasAdddense_73/MatMul:product:0'dense_73/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: Y
dense_73/ReluReludense_73/BiasAdd:output:0*
T0*
_output_shapes

: 
dense_74/MatMul/ReadVariableOpReadVariableOp'dense_74_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_74/MatMulMatMuldense_73/Relu:activations:0&dense_74/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_74/BiasAdd/ReadVariableOpReadVariableOp(dense_74_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_74/BiasAddBiasAdddense_74/MatMul:product:0'dense_74/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: Y
dense_74/ReluReludense_74/BiasAdd:output:0*
T0*
_output_shapes

: 
dense_75/MatMul/ReadVariableOpReadVariableOp'dense_75_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_75/MatMulMatMuldense_74/Relu:activations:0&dense_75/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_75/BiasAdd/ReadVariableOpReadVariableOp(dense_75_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_75/BiasAddBiasAdddense_75/MatMul:product:0'dense_75/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: Y
dense_75/ReluReludense_75/BiasAdd:output:0*
T0*
_output_shapes

: 
dense_76/MatMul/ReadVariableOpReadVariableOp'dense_76_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_76/MatMulMatMuldense_75/Relu:activations:0&dense_76/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_76/BiasAdd/ReadVariableOpReadVariableOp(dense_76_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_76/BiasAddBiasAdddense_76/MatMul:product:0'dense_76/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_
IdentityIdentitydense_76/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp ^dense_72/BiasAdd/ReadVariableOp^dense_72/MatMul/ReadVariableOp ^dense_73/BiasAdd/ReadVariableOp^dense_73/MatMul/ReadVariableOp ^dense_74/BiasAdd/ReadVariableOp^dense_74/MatMul/ReadVariableOp ^dense_75/BiasAdd/ReadVariableOp^dense_75/MatMul/ReadVariableOp ^dense_76/BiasAdd/ReadVariableOp^dense_76/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2B
dense_72/BiasAdd/ReadVariableOpdense_72/BiasAdd/ReadVariableOp2@
dense_72/MatMul/ReadVariableOpdense_72/MatMul/ReadVariableOp2B
dense_73/BiasAdd/ReadVariableOpdense_73/BiasAdd/ReadVariableOp2@
dense_73/MatMul/ReadVariableOpdense_73/MatMul/ReadVariableOp2B
dense_74/BiasAdd/ReadVariableOpdense_74/BiasAdd/ReadVariableOp2@
dense_74/MatMul/ReadVariableOpdense_74/MatMul/ReadVariableOp2B
dense_75/BiasAdd/ReadVariableOpdense_75/BiasAdd/ReadVariableOp2@
dense_75/MatMul/ReadVariableOpdense_75/MatMul/ReadVariableOp2B
dense_76/BiasAdd/ReadVariableOpdense_76/BiasAdd/ReadVariableOp2@
dense_76/MatMul/ReadVariableOpdense_76/MatMul/ReadVariableOp:J F
"
_output_shapes
:
 
_user_specified_nameinputs
?

*__inference_dense_72_layer_call_fn_1021610

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

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_72_layer_call_and_return_conditional_losses_1021109f
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
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs

๔
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021319

inputs"
dense_72_1021293:
dense_72_1021295:"
dense_73_1021298: 
dense_73_1021300: "
dense_74_1021303:  
dense_74_1021305: "
dense_75_1021308:  
dense_75_1021310: "
dense_76_1021313: 
dense_76_1021315:
identityข dense_72/StatefulPartitionedCallข dense_73/StatefulPartitionedCallข dense_74/StatefulPartitionedCallข dense_75/StatefulPartitionedCallข dense_76/StatefulPartitionedCallด
flatten_20/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_20_layer_call_and_return_conditional_losses_1021096
 dense_72/StatefulPartitionedCallStatefulPartitionedCall#flatten_20/PartitionedCall:output:0dense_72_1021293dense_72_1021295*
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
GPU 2J 8 *N
fIRG
E__inference_dense_72_layer_call_and_return_conditional_losses_1021109
 dense_73/StatefulPartitionedCallStatefulPartitionedCall)dense_72/StatefulPartitionedCall:output:0dense_73_1021298dense_73_1021300*
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
GPU 2J 8 *N
fIRG
E__inference_dense_73_layer_call_and_return_conditional_losses_1021126
 dense_74/StatefulPartitionedCallStatefulPartitionedCall)dense_73/StatefulPartitionedCall:output:0dense_74_1021303dense_74_1021305*
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
GPU 2J 8 *N
fIRG
E__inference_dense_74_layer_call_and_return_conditional_losses_1021143
 dense_75/StatefulPartitionedCallStatefulPartitionedCall)dense_74/StatefulPartitionedCall:output:0dense_75_1021308dense_75_1021310*
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
GPU 2J 8 *N
fIRG
E__inference_dense_75_layer_call_and_return_conditional_losses_1021160
 dense_76/StatefulPartitionedCallStatefulPartitionedCall)dense_75/StatefulPartitionedCall:output:0dense_76_1021313dense_76_1021315*
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
GPU 2J 8 *N
fIRG
E__inference_dense_76_layer_call_and_return_conditional_losses_1021176o
IdentityIdentity)dense_76/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:๕
NoOpNoOp!^dense_72/StatefulPartitionedCall!^dense_73/StatefulPartitionedCall!^dense_74/StatefulPartitionedCall!^dense_75/StatefulPartitionedCall!^dense_76/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2D
 dense_73/StatefulPartitionedCall dense_73/StatefulPartitionedCall2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall2D
 dense_75/StatefulPartitionedCall dense_75/StatefulPartitionedCall2D
 dense_76/StatefulPartitionedCall dense_76/StatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
ๆ	
๖
E__inference_dense_74_layer_call_and_return_conditional_losses_1021143

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

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

: X
IdentityIdentityRelu:activations:0^NoOp*
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

H
,__inference_flatten_20_layer_call_fn_1021595

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

:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_20_layer_call_and_return_conditional_losses_1021096W
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs


?
7__inference_Flatten5Layers_batch8_layer_call_fn_1021367
input_29
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
StatefulPartitionedCallStatefulPartitionedCallinput_29unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
GPU 2J 8 *[
fVRT
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021319f
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
":: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
"
_output_shapes
:
"
_user_specified_name
input_29
?

*__inference_dense_75_layer_call_fn_1021670

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

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_75_layer_call_and_return_conditional_losses_1021160f
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

c
G__inference_flatten_20_layer_call_and_return_conditional_losses_1021601

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

:O
IdentityIdentityReshape:output:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs

c
G__inference_flatten_20_layer_call_and_return_conditional_losses_1021096

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

:O
IdentityIdentityReshape:output:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
?

*__inference_dense_73_layer_call_fn_1021630

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

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_73_layer_call_and_return_conditional_losses_1021126f
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
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs
ๆ	
๖
E__inference_dense_73_layer_call_and_return_conditional_losses_1021126

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

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

: X
IdentityIdentityRelu:activations:0^NoOp*
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
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
แM
ง
 __inference__traced_save_1021834
file_prefix.
*savev2_dense_72_kernel_read_readvariableop,
(savev2_dense_72_bias_read_readvariableop.
*savev2_dense_73_kernel_read_readvariableop,
(savev2_dense_73_bias_read_readvariableop.
*savev2_dense_74_kernel_read_readvariableop,
(savev2_dense_74_bias_read_readvariableop.
*savev2_dense_75_kernel_read_readvariableop,
(savev2_dense_75_bias_read_readvariableop.
*savev2_dense_76_kernel_read_readvariableop,
(savev2_dense_76_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_72_kernel_m_read_readvariableop3
/savev2_adam_dense_72_bias_m_read_readvariableop5
1savev2_adam_dense_73_kernel_m_read_readvariableop3
/savev2_adam_dense_73_bias_m_read_readvariableop5
1savev2_adam_dense_74_kernel_m_read_readvariableop3
/savev2_adam_dense_74_bias_m_read_readvariableop5
1savev2_adam_dense_75_kernel_m_read_readvariableop3
/savev2_adam_dense_75_bias_m_read_readvariableop5
1savev2_adam_dense_76_kernel_m_read_readvariableop3
/savev2_adam_dense_76_bias_m_read_readvariableop5
1savev2_adam_dense_72_kernel_v_read_readvariableop3
/savev2_adam_dense_72_bias_v_read_readvariableop5
1savev2_adam_dense_73_kernel_v_read_readvariableop3
/savev2_adam_dense_73_bias_v_read_readvariableop5
1savev2_adam_dense_74_kernel_v_read_readvariableop3
/savev2_adam_dense_74_bias_v_read_readvariableop5
1savev2_adam_dense_75_kernel_v_read_readvariableop3
/savev2_adam_dense_75_bias_v_read_readvariableop5
1savev2_adam_dense_76_kernel_v_read_readvariableop3
/savev2_adam_dense_76_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_72_kernel_read_readvariableop(savev2_dense_72_bias_read_readvariableop*savev2_dense_73_kernel_read_readvariableop(savev2_dense_73_bias_read_readvariableop*savev2_dense_74_kernel_read_readvariableop(savev2_dense_74_bias_read_readvariableop*savev2_dense_75_kernel_read_readvariableop(savev2_dense_75_bias_read_readvariableop*savev2_dense_76_kernel_read_readvariableop(savev2_dense_76_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_72_kernel_m_read_readvariableop/savev2_adam_dense_72_bias_m_read_readvariableop1savev2_adam_dense_73_kernel_m_read_readvariableop/savev2_adam_dense_73_bias_m_read_readvariableop1savev2_adam_dense_74_kernel_m_read_readvariableop/savev2_adam_dense_74_bias_m_read_readvariableop1savev2_adam_dense_75_kernel_m_read_readvariableop/savev2_adam_dense_75_bias_m_read_readvariableop1savev2_adam_dense_76_kernel_m_read_readvariableop/savev2_adam_dense_76_bias_m_read_readvariableop1savev2_adam_dense_72_kernel_v_read_readvariableop/savev2_adam_dense_72_bias_v_read_readvariableop1savev2_adam_dense_73_kernel_v_read_readvariableop/savev2_adam_dense_73_bias_v_read_readvariableop1savev2_adam_dense_74_kernel_v_read_readvariableop/savev2_adam_dense_74_bias_v_read_readvariableop1savev2_adam_dense_75_kernel_v_read_readvariableop/savev2_adam_dense_75_bias_v_read_readvariableop1savev2_adam_dense_76_kernel_v_read_readvariableop/savev2_adam_dense_76_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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


?
7__inference_Flatten5Layers_batch8_layer_call_fn_1021206
input_29
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
StatefulPartitionedCallStatefulPartitionedCallinput_29unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
GPU 2J 8 *[
fVRT
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021183f
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
":: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
"
_output_shapes
:
"
_user_specified_name
input_29
	
๖
E__inference_dense_76_layer_call_and_return_conditional_losses_1021700

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
ุ,
?
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021550

inputs9
'dense_72_matmul_readvariableop_resource:6
(dense_72_biasadd_readvariableop_resource:9
'dense_73_matmul_readvariableop_resource: 6
(dense_73_biasadd_readvariableop_resource: 9
'dense_74_matmul_readvariableop_resource:  6
(dense_74_biasadd_readvariableop_resource: 9
'dense_75_matmul_readvariableop_resource:  6
(dense_75_biasadd_readvariableop_resource: 9
'dense_76_matmul_readvariableop_resource: 6
(dense_76_biasadd_readvariableop_resource:
identityขdense_72/BiasAdd/ReadVariableOpขdense_72/MatMul/ReadVariableOpขdense_73/BiasAdd/ReadVariableOpขdense_73/MatMul/ReadVariableOpขdense_74/BiasAdd/ReadVariableOpขdense_74/MatMul/ReadVariableOpขdense_75/BiasAdd/ReadVariableOpขdense_75/MatMul/ReadVariableOpขdense_76/BiasAdd/ReadVariableOpขdense_76/MatMul/ReadVariableOpa
flatten_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   i
flatten_20/ReshapeReshapeinputsflatten_20/Const:output:0*
T0*
_output_shapes

:
dense_72/MatMul/ReadVariableOpReadVariableOp'dense_72_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_72/MatMulMatMulflatten_20/Reshape:output:0&dense_72/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_72/BiasAdd/ReadVariableOpReadVariableOp(dense_72_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_72/BiasAddBiasAdddense_72/MatMul:product:0'dense_72/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Y
dense_72/ReluReludense_72/BiasAdd:output:0*
T0*
_output_shapes

:
dense_73/MatMul/ReadVariableOpReadVariableOp'dense_73_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_73/MatMulMatMuldense_72/Relu:activations:0&dense_73/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_73/BiasAdd/ReadVariableOpReadVariableOp(dense_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_73/BiasAddBiasAdddense_73/MatMul:product:0'dense_73/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: Y
dense_73/ReluReludense_73/BiasAdd:output:0*
T0*
_output_shapes

: 
dense_74/MatMul/ReadVariableOpReadVariableOp'dense_74_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_74/MatMulMatMuldense_73/Relu:activations:0&dense_74/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_74/BiasAdd/ReadVariableOpReadVariableOp(dense_74_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_74/BiasAddBiasAdddense_74/MatMul:product:0'dense_74/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: Y
dense_74/ReluReludense_74/BiasAdd:output:0*
T0*
_output_shapes

: 
dense_75/MatMul/ReadVariableOpReadVariableOp'dense_75_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_75/MatMulMatMuldense_74/Relu:activations:0&dense_75/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
dense_75/BiasAdd/ReadVariableOpReadVariableOp(dense_75_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_75/BiasAddBiasAdddense_75/MatMul:product:0'dense_75/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: Y
dense_75/ReluReludense_75/BiasAdd:output:0*
T0*
_output_shapes

: 
dense_76/MatMul/ReadVariableOpReadVariableOp'dense_76_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_76/MatMulMatMuldense_75/Relu:activations:0&dense_76/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_76/BiasAdd/ReadVariableOpReadVariableOp(dense_76_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_76/BiasAddBiasAdddense_76/MatMul:product:0'dense_76/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_
IdentityIdentitydense_76/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp ^dense_72/BiasAdd/ReadVariableOp^dense_72/MatMul/ReadVariableOp ^dense_73/BiasAdd/ReadVariableOp^dense_73/MatMul/ReadVariableOp ^dense_74/BiasAdd/ReadVariableOp^dense_74/MatMul/ReadVariableOp ^dense_75/BiasAdd/ReadVariableOp^dense_75/MatMul/ReadVariableOp ^dense_76/BiasAdd/ReadVariableOp^dense_76/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2B
dense_72/BiasAdd/ReadVariableOpdense_72/BiasAdd/ReadVariableOp2@
dense_72/MatMul/ReadVariableOpdense_72/MatMul/ReadVariableOp2B
dense_73/BiasAdd/ReadVariableOpdense_73/BiasAdd/ReadVariableOp2@
dense_73/MatMul/ReadVariableOpdense_73/MatMul/ReadVariableOp2B
dense_74/BiasAdd/ReadVariableOpdense_74/BiasAdd/ReadVariableOp2@
dense_74/MatMul/ReadVariableOpdense_74/MatMul/ReadVariableOp2B
dense_75/BiasAdd/ReadVariableOpdense_75/BiasAdd/ReadVariableOp2@
dense_75/MatMul/ReadVariableOpdense_75/MatMul/ReadVariableOp2B
dense_76/BiasAdd/ReadVariableOpdense_76/BiasAdd/ReadVariableOp2@
dense_76/MatMul/ReadVariableOpdense_76/MatMul/ReadVariableOp:J F
"
_output_shapes
:
 
_user_specified_nameinputs

๖
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021397
input_29"
dense_72_1021371:
dense_72_1021373:"
dense_73_1021376: 
dense_73_1021378: "
dense_74_1021381:  
dense_74_1021383: "
dense_75_1021386:  
dense_75_1021388: "
dense_76_1021391: 
dense_76_1021393:
identityข dense_72/StatefulPartitionedCallข dense_73/StatefulPartitionedCallข dense_74/StatefulPartitionedCallข dense_75/StatefulPartitionedCallข dense_76/StatefulPartitionedCallถ
flatten_20/PartitionedCallPartitionedCallinput_29*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_20_layer_call_and_return_conditional_losses_1021096
 dense_72/StatefulPartitionedCallStatefulPartitionedCall#flatten_20/PartitionedCall:output:0dense_72_1021371dense_72_1021373*
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
GPU 2J 8 *N
fIRG
E__inference_dense_72_layer_call_and_return_conditional_losses_1021109
 dense_73/StatefulPartitionedCallStatefulPartitionedCall)dense_72/StatefulPartitionedCall:output:0dense_73_1021376dense_73_1021378*
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
GPU 2J 8 *N
fIRG
E__inference_dense_73_layer_call_and_return_conditional_losses_1021126
 dense_74/StatefulPartitionedCallStatefulPartitionedCall)dense_73/StatefulPartitionedCall:output:0dense_74_1021381dense_74_1021383*
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
GPU 2J 8 *N
fIRG
E__inference_dense_74_layer_call_and_return_conditional_losses_1021143
 dense_75/StatefulPartitionedCallStatefulPartitionedCall)dense_74/StatefulPartitionedCall:output:0dense_75_1021386dense_75_1021388*
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
GPU 2J 8 *N
fIRG
E__inference_dense_75_layer_call_and_return_conditional_losses_1021160
 dense_76/StatefulPartitionedCallStatefulPartitionedCall)dense_75/StatefulPartitionedCall:output:0dense_76_1021391dense_76_1021393*
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
GPU 2J 8 *N
fIRG
E__inference_dense_76_layer_call_and_return_conditional_losses_1021176o
IdentityIdentity)dense_76/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:๕
NoOpNoOp!^dense_72/StatefulPartitionedCall!^dense_73/StatefulPartitionedCall!^dense_74/StatefulPartitionedCall!^dense_75/StatefulPartitionedCall!^dense_76/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2D
 dense_73/StatefulPartitionedCall dense_73/StatefulPartitionedCall2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall2D
 dense_75/StatefulPartitionedCall dense_75/StatefulPartitionedCall2D
 dense_76/StatefulPartitionedCall dense_76/StatefulPartitionedCall:L H
"
_output_shapes
:
"
_user_specified_name
input_29
ึ	
์
%__inference_signature_wrapper_1021460
input_29
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
StatefulPartitionedCallStatefulPartitionedCallinput_29unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
"__inference__wrapped_model_1021083f
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
":: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
"
_output_shapes
:
"
_user_specified_name
input_29
ๆ	
๖
E__inference_dense_75_layer_call_and_return_conditional_losses_1021681

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

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

: X
IdentityIdentityRelu:activations:0^NoOp*
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
ๆ	
๖
E__inference_dense_73_layer_call_and_return_conditional_losses_1021641

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

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

: X
IdentityIdentityRelu:activations:0^NoOp*
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
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs

๔
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021183

inputs"
dense_72_1021110:
dense_72_1021112:"
dense_73_1021127: 
dense_73_1021129: "
dense_74_1021144:  
dense_74_1021146: "
dense_75_1021161:  
dense_75_1021163: "
dense_76_1021177: 
dense_76_1021179:
identityข dense_72/StatefulPartitionedCallข dense_73/StatefulPartitionedCallข dense_74/StatefulPartitionedCallข dense_75/StatefulPartitionedCallข dense_76/StatefulPartitionedCallด
flatten_20/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_20_layer_call_and_return_conditional_losses_1021096
 dense_72/StatefulPartitionedCallStatefulPartitionedCall#flatten_20/PartitionedCall:output:0dense_72_1021110dense_72_1021112*
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
GPU 2J 8 *N
fIRG
E__inference_dense_72_layer_call_and_return_conditional_losses_1021109
 dense_73/StatefulPartitionedCallStatefulPartitionedCall)dense_72/StatefulPartitionedCall:output:0dense_73_1021127dense_73_1021129*
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
GPU 2J 8 *N
fIRG
E__inference_dense_73_layer_call_and_return_conditional_losses_1021126
 dense_74/StatefulPartitionedCallStatefulPartitionedCall)dense_73/StatefulPartitionedCall:output:0dense_74_1021144dense_74_1021146*
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
GPU 2J 8 *N
fIRG
E__inference_dense_74_layer_call_and_return_conditional_losses_1021143
 dense_75/StatefulPartitionedCallStatefulPartitionedCall)dense_74/StatefulPartitionedCall:output:0dense_75_1021161dense_75_1021163*
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
GPU 2J 8 *N
fIRG
E__inference_dense_75_layer_call_and_return_conditional_losses_1021160
 dense_76/StatefulPartitionedCallStatefulPartitionedCall)dense_75/StatefulPartitionedCall:output:0dense_76_1021177dense_76_1021179*
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
GPU 2J 8 *N
fIRG
E__inference_dense_76_layer_call_and_return_conditional_losses_1021176o
IdentityIdentity)dense_76/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:๕
NoOpNoOp!^dense_72/StatefulPartitionedCall!^dense_73/StatefulPartitionedCall!^dense_74/StatefulPartitionedCall!^dense_75/StatefulPartitionedCall!^dense_76/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2D
 dense_73/StatefulPartitionedCall dense_73/StatefulPartitionedCall2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall2D
 dense_75/StatefulPartitionedCall dense_75/StatefulPartitionedCall2D
 dense_76/StatefulPartitionedCall dense_76/StatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
ๆ	
๖
E__inference_dense_72_layer_call_and_return_conditional_losses_1021621

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

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:X
IdentityIdentityRelu:activations:0^NoOp*
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
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
?

*__inference_dense_76_layer_call_fn_1021690

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

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_76_layer_call_and_return_conditional_losses_1021176f
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

๖
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021427
input_29"
dense_72_1021401:
dense_72_1021403:"
dense_73_1021406: 
dense_73_1021408: "
dense_74_1021411:  
dense_74_1021413: "
dense_75_1021416:  
dense_75_1021418: "
dense_76_1021421: 
dense_76_1021423:
identityข dense_72/StatefulPartitionedCallข dense_73/StatefulPartitionedCallข dense_74/StatefulPartitionedCallข dense_75/StatefulPartitionedCallข dense_76/StatefulPartitionedCallถ
flatten_20/PartitionedCallPartitionedCallinput_29*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_20_layer_call_and_return_conditional_losses_1021096
 dense_72/StatefulPartitionedCallStatefulPartitionedCall#flatten_20/PartitionedCall:output:0dense_72_1021401dense_72_1021403*
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
GPU 2J 8 *N
fIRG
E__inference_dense_72_layer_call_and_return_conditional_losses_1021109
 dense_73/StatefulPartitionedCallStatefulPartitionedCall)dense_72/StatefulPartitionedCall:output:0dense_73_1021406dense_73_1021408*
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
GPU 2J 8 *N
fIRG
E__inference_dense_73_layer_call_and_return_conditional_losses_1021126
 dense_74/StatefulPartitionedCallStatefulPartitionedCall)dense_73/StatefulPartitionedCall:output:0dense_74_1021411dense_74_1021413*
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
GPU 2J 8 *N
fIRG
E__inference_dense_74_layer_call_and_return_conditional_losses_1021143
 dense_75/StatefulPartitionedCallStatefulPartitionedCall)dense_74/StatefulPartitionedCall:output:0dense_75_1021416dense_75_1021418*
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
GPU 2J 8 *N
fIRG
E__inference_dense_75_layer_call_and_return_conditional_losses_1021160
 dense_76/StatefulPartitionedCallStatefulPartitionedCall)dense_75/StatefulPartitionedCall:output:0dense_76_1021421dense_76_1021423*
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
GPU 2J 8 *N
fIRG
E__inference_dense_76_layer_call_and_return_conditional_losses_1021176o
IdentityIdentity)dense_76/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:๕
NoOpNoOp!^dense_72/StatefulPartitionedCall!^dense_73/StatefulPartitionedCall!^dense_74/StatefulPartitionedCall!^dense_75/StatefulPartitionedCall!^dense_76/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2D
 dense_73/StatefulPartitionedCall dense_73/StatefulPartitionedCall2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall2D
 dense_75/StatefulPartitionedCall dense_75/StatefulPartitionedCall2D
 dense_76/StatefulPartitionedCall dense_76/StatefulPartitionedCall:L H
"
_output_shapes
:
"
_user_specified_name
input_29
ๆ	
๖
E__inference_dense_75_layer_call_and_return_conditional_losses_1021160

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

: r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

: X
IdentityIdentityRelu:activations:0^NoOp*
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
	
๖
E__inference_dense_76_layer_call_and_return_conditional_losses_1021176

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
?

*__inference_dense_74_layer_call_fn_1021650

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

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_74_layer_call_and_return_conditional_losses_1021143f
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


?
7__inference_Flatten5Layers_batch8_layer_call_fn_1021485

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

:*,
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
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021183f
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
":: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
ำ?

"__inference__wrapped_model_1021083
input_29O
=flatten5layers_batch8_dense_72_matmul_readvariableop_resource:L
>flatten5layers_batch8_dense_72_biasadd_readvariableop_resource:O
=flatten5layers_batch8_dense_73_matmul_readvariableop_resource: L
>flatten5layers_batch8_dense_73_biasadd_readvariableop_resource: O
=flatten5layers_batch8_dense_74_matmul_readvariableop_resource:  L
>flatten5layers_batch8_dense_74_biasadd_readvariableop_resource: O
=flatten5layers_batch8_dense_75_matmul_readvariableop_resource:  L
>flatten5layers_batch8_dense_75_biasadd_readvariableop_resource: O
=flatten5layers_batch8_dense_76_matmul_readvariableop_resource: L
>flatten5layers_batch8_dense_76_biasadd_readvariableop_resource:
identityข5Flatten5Layers_batch8/dense_72/BiasAdd/ReadVariableOpข4Flatten5Layers_batch8/dense_72/MatMul/ReadVariableOpข5Flatten5Layers_batch8/dense_73/BiasAdd/ReadVariableOpข4Flatten5Layers_batch8/dense_73/MatMul/ReadVariableOpข5Flatten5Layers_batch8/dense_74/BiasAdd/ReadVariableOpข4Flatten5Layers_batch8/dense_74/MatMul/ReadVariableOpข5Flatten5Layers_batch8/dense_75/BiasAdd/ReadVariableOpข4Flatten5Layers_batch8/dense_75/MatMul/ReadVariableOpข5Flatten5Layers_batch8/dense_76/BiasAdd/ReadVariableOpข4Flatten5Layers_batch8/dense_76/MatMul/ReadVariableOpw
&Flatten5Layers_batch8/flatten_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   
(Flatten5Layers_batch8/flatten_20/ReshapeReshapeinput_29/Flatten5Layers_batch8/flatten_20/Const:output:0*
T0*
_output_shapes

:ฒ
4Flatten5Layers_batch8/dense_72/MatMul/ReadVariableOpReadVariableOp=flatten5layers_batch8_dense_72_matmul_readvariableop_resource*
_output_shapes

:*
dtype0ษ
%Flatten5Layers_batch8/dense_72/MatMulMatMul1Flatten5Layers_batch8/flatten_20/Reshape:output:0<Flatten5Layers_batch8/dense_72/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:ฐ
5Flatten5Layers_batch8/dense_72/BiasAdd/ReadVariableOpReadVariableOp>flatten5layers_batch8_dense_72_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ส
&Flatten5Layers_batch8/dense_72/BiasAddBiasAdd/Flatten5Layers_batch8/dense_72/MatMul:product:0=Flatten5Layers_batch8/dense_72/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
#Flatten5Layers_batch8/dense_72/ReluRelu/Flatten5Layers_batch8/dense_72/BiasAdd:output:0*
T0*
_output_shapes

:ฒ
4Flatten5Layers_batch8/dense_73/MatMul/ReadVariableOpReadVariableOp=flatten5layers_batch8_dense_73_matmul_readvariableop_resource*
_output_shapes

: *
dtype0ษ
%Flatten5Layers_batch8/dense_73/MatMulMatMul1Flatten5Layers_batch8/dense_72/Relu:activations:0<Flatten5Layers_batch8/dense_73/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ฐ
5Flatten5Layers_batch8/dense_73/BiasAdd/ReadVariableOpReadVariableOp>flatten5layers_batch8_dense_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ส
&Flatten5Layers_batch8/dense_73/BiasAddBiasAdd/Flatten5Layers_batch8/dense_73/MatMul:product:0=Flatten5Layers_batch8/dense_73/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
#Flatten5Layers_batch8/dense_73/ReluRelu/Flatten5Layers_batch8/dense_73/BiasAdd:output:0*
T0*
_output_shapes

: ฒ
4Flatten5Layers_batch8/dense_74/MatMul/ReadVariableOpReadVariableOp=flatten5layers_batch8_dense_74_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0ษ
%Flatten5Layers_batch8/dense_74/MatMulMatMul1Flatten5Layers_batch8/dense_73/Relu:activations:0<Flatten5Layers_batch8/dense_74/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ฐ
5Flatten5Layers_batch8/dense_74/BiasAdd/ReadVariableOpReadVariableOp>flatten5layers_batch8_dense_74_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ส
&Flatten5Layers_batch8/dense_74/BiasAddBiasAdd/Flatten5Layers_batch8/dense_74/MatMul:product:0=Flatten5Layers_batch8/dense_74/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
#Flatten5Layers_batch8/dense_74/ReluRelu/Flatten5Layers_batch8/dense_74/BiasAdd:output:0*
T0*
_output_shapes

: ฒ
4Flatten5Layers_batch8/dense_75/MatMul/ReadVariableOpReadVariableOp=flatten5layers_batch8_dense_75_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0ษ
%Flatten5Layers_batch8/dense_75/MatMulMatMul1Flatten5Layers_batch8/dense_74/Relu:activations:0<Flatten5Layers_batch8/dense_75/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ฐ
5Flatten5Layers_batch8/dense_75/BiasAdd/ReadVariableOpReadVariableOp>flatten5layers_batch8_dense_75_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ส
&Flatten5Layers_batch8/dense_75/BiasAddBiasAdd/Flatten5Layers_batch8/dense_75/MatMul:product:0=Flatten5Layers_batch8/dense_75/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
#Flatten5Layers_batch8/dense_75/ReluRelu/Flatten5Layers_batch8/dense_75/BiasAdd:output:0*
T0*
_output_shapes

: ฒ
4Flatten5Layers_batch8/dense_76/MatMul/ReadVariableOpReadVariableOp=flatten5layers_batch8_dense_76_matmul_readvariableop_resource*
_output_shapes

: *
dtype0ษ
%Flatten5Layers_batch8/dense_76/MatMulMatMul1Flatten5Layers_batch8/dense_75/Relu:activations:0<Flatten5Layers_batch8/dense_76/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:ฐ
5Flatten5Layers_batch8/dense_76/BiasAdd/ReadVariableOpReadVariableOp>flatten5layers_batch8_dense_76_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ส
&Flatten5Layers_batch8/dense_76/BiasAddBiasAdd/Flatten5Layers_batch8/dense_76/MatMul:product:0=Flatten5Layers_batch8/dense_76/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:u
IdentityIdentity/Flatten5Layers_batch8/dense_76/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:๑
NoOpNoOp6^Flatten5Layers_batch8/dense_72/BiasAdd/ReadVariableOp5^Flatten5Layers_batch8/dense_72/MatMul/ReadVariableOp6^Flatten5Layers_batch8/dense_73/BiasAdd/ReadVariableOp5^Flatten5Layers_batch8/dense_73/MatMul/ReadVariableOp6^Flatten5Layers_batch8/dense_74/BiasAdd/ReadVariableOp5^Flatten5Layers_batch8/dense_74/MatMul/ReadVariableOp6^Flatten5Layers_batch8/dense_75/BiasAdd/ReadVariableOp5^Flatten5Layers_batch8/dense_75/MatMul/ReadVariableOp6^Flatten5Layers_batch8/dense_76/BiasAdd/ReadVariableOp5^Flatten5Layers_batch8/dense_76/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : 2n
5Flatten5Layers_batch8/dense_72/BiasAdd/ReadVariableOp5Flatten5Layers_batch8/dense_72/BiasAdd/ReadVariableOp2l
4Flatten5Layers_batch8/dense_72/MatMul/ReadVariableOp4Flatten5Layers_batch8/dense_72/MatMul/ReadVariableOp2n
5Flatten5Layers_batch8/dense_73/BiasAdd/ReadVariableOp5Flatten5Layers_batch8/dense_73/BiasAdd/ReadVariableOp2l
4Flatten5Layers_batch8/dense_73/MatMul/ReadVariableOp4Flatten5Layers_batch8/dense_73/MatMul/ReadVariableOp2n
5Flatten5Layers_batch8/dense_74/BiasAdd/ReadVariableOp5Flatten5Layers_batch8/dense_74/BiasAdd/ReadVariableOp2l
4Flatten5Layers_batch8/dense_74/MatMul/ReadVariableOp4Flatten5Layers_batch8/dense_74/MatMul/ReadVariableOp2n
5Flatten5Layers_batch8/dense_75/BiasAdd/ReadVariableOp5Flatten5Layers_batch8/dense_75/BiasAdd/ReadVariableOp2l
4Flatten5Layers_batch8/dense_75/MatMul/ReadVariableOp4Flatten5Layers_batch8/dense_75/MatMul/ReadVariableOp2n
5Flatten5Layers_batch8/dense_76/BiasAdd/ReadVariableOp5Flatten5Layers_batch8/dense_76/BiasAdd/ReadVariableOp2l
4Flatten5Layers_batch8/dense_76/MatMul/ReadVariableOp4Flatten5Layers_batch8/dense_76/MatMul/ReadVariableOp:L H
"
_output_shapes
:
"
_user_specified_name
input_29


?
7__inference_Flatten5Layers_batch8_layer_call_fn_1021510

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

:*,
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
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021319f
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
":: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
ๆ	
๖
E__inference_dense_72_layer_call_and_return_conditional_losses_1021109

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

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:X
IdentityIdentityRelu:activations:0^NoOp*
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
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
ภ
ฉ
#__inference__traced_restore_1021955
file_prefix2
 assignvariableop_dense_72_kernel:.
 assignvariableop_1_dense_72_bias:4
"assignvariableop_2_dense_73_kernel: .
 assignvariableop_3_dense_73_bias: 4
"assignvariableop_4_dense_74_kernel:  .
 assignvariableop_5_dense_74_bias: 4
"assignvariableop_6_dense_75_kernel:  .
 assignvariableop_7_dense_75_bias: 4
"assignvariableop_8_dense_76_kernel: .
 assignvariableop_9_dense_76_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: #
assignvariableop_15_total: #
assignvariableop_16_count: <
*assignvariableop_17_adam_dense_72_kernel_m:6
(assignvariableop_18_adam_dense_72_bias_m:<
*assignvariableop_19_adam_dense_73_kernel_m: 6
(assignvariableop_20_adam_dense_73_bias_m: <
*assignvariableop_21_adam_dense_74_kernel_m:  6
(assignvariableop_22_adam_dense_74_bias_m: <
*assignvariableop_23_adam_dense_75_kernel_m:  6
(assignvariableop_24_adam_dense_75_bias_m: <
*assignvariableop_25_adam_dense_76_kernel_m: 6
(assignvariableop_26_adam_dense_76_bias_m:<
*assignvariableop_27_adam_dense_72_kernel_v:6
(assignvariableop_28_adam_dense_72_bias_v:<
*assignvariableop_29_adam_dense_73_kernel_v: 6
(assignvariableop_30_adam_dense_73_bias_v: <
*assignvariableop_31_adam_dense_74_kernel_v:  6
(assignvariableop_32_adam_dense_74_bias_v: <
*assignvariableop_33_adam_dense_75_kernel_v:  6
(assignvariableop_34_adam_dense_75_bias_v: <
*assignvariableop_35_adam_dense_76_kernel_v: 6
(assignvariableop_36_adam_dense_76_bias_v:
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
AssignVariableOpAssignVariableOp assignvariableop_dense_72_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_72_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_73_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_73_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_74_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_74_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_75_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_75_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_76_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_76_biasIdentity_9:output:0"/device:CPU:0*
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
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_72_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_72_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_73_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_73_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_74_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_74_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_75_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_75_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_76_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_76_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_72_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_72_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_73_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_73_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_74_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_74_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_75_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_75_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_76_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_76_bias_vIdentity_36:output:0"/device:CPU:0*
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
_user_specified_namefile_prefix"ต	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
8
input_29,
serving_default_input_29:03
dense_76'
StatefulPartitionedCall:0tensorflow/serving/predict:ก
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
7__inference_Flatten5Layers_batch8_layer_call_fn_1021206
7__inference_Flatten5Layers_batch8_layer_call_fn_1021485
7__inference_Flatten5Layers_batch8_layer_call_fn_1021510
7__inference_Flatten5Layers_batch8_layer_call_fn_1021367ฟ
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
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021550
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021590
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021397
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021427ฟ
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
"__inference__wrapped_model_1021083input_29"
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
,__inference_flatten_20_layer_call_fn_1021595ข
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
G__inference_flatten_20_layer_call_and_return_conditional_losses_1021601ข
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
*__inference_dense_72_layer_call_fn_1021610ข
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
E__inference_dense_72_layer_call_and_return_conditional_losses_1021621ข
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
!:2dense_72/kernel
:2dense_72/bias
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
*__inference_dense_73_layer_call_fn_1021630ข
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
E__inference_dense_73_layer_call_and_return_conditional_losses_1021641ข
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
!: 2dense_73/kernel
: 2dense_73/bias
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
*__inference_dense_74_layer_call_fn_1021650ข
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
E__inference_dense_74_layer_call_and_return_conditional_losses_1021661ข
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
!:  2dense_74/kernel
: 2dense_74/bias
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
*__inference_dense_75_layer_call_fn_1021670ข
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
E__inference_dense_75_layer_call_and_return_conditional_losses_1021681ข
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
!:  2dense_75/kernel
: 2dense_75/bias
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
*__inference_dense_76_layer_call_fn_1021690ข
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
E__inference_dense_76_layer_call_and_return_conditional_losses_1021700ข
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
!: 2dense_76/kernel
:2dense_76/bias
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
7__inference_Flatten5Layers_batch8_layer_call_fn_1021206input_29"ฟ
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
7__inference_Flatten5Layers_batch8_layer_call_fn_1021485inputs"ฟ
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
7__inference_Flatten5Layers_batch8_layer_call_fn_1021510inputs"ฟ
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
7__inference_Flatten5Layers_batch8_layer_call_fn_1021367input_29"ฟ
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
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021550inputs"ฟ
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
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021590inputs"ฟ
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
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021397input_29"ฟ
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
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021427input_29"ฟ
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
%__inference_signature_wrapper_1021460input_29"
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
,__inference_flatten_20_layer_call_fn_1021595inputs"ข
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
G__inference_flatten_20_layer_call_and_return_conditional_losses_1021601inputs"ข
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
*__inference_dense_72_layer_call_fn_1021610inputs"ข
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
E__inference_dense_72_layer_call_and_return_conditional_losses_1021621inputs"ข
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
*__inference_dense_73_layer_call_fn_1021630inputs"ข
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
E__inference_dense_73_layer_call_and_return_conditional_losses_1021641inputs"ข
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
*__inference_dense_74_layer_call_fn_1021650inputs"ข
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
E__inference_dense_74_layer_call_and_return_conditional_losses_1021661inputs"ข
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
*__inference_dense_75_layer_call_fn_1021670inputs"ข
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
E__inference_dense_75_layer_call_and_return_conditional_losses_1021681inputs"ข
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
*__inference_dense_76_layer_call_fn_1021690inputs"ข
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
E__inference_dense_76_layer_call_and_return_conditional_losses_1021700inputs"ข
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
&:$2Adam/dense_72/kernel/m
 :2Adam/dense_72/bias/m
&:$ 2Adam/dense_73/kernel/m
 : 2Adam/dense_73/bias/m
&:$  2Adam/dense_74/kernel/m
 : 2Adam/dense_74/bias/m
&:$  2Adam/dense_75/kernel/m
 : 2Adam/dense_75/bias/m
&:$ 2Adam/dense_76/kernel/m
 :2Adam/dense_76/bias/m
&:$2Adam/dense_72/kernel/v
 :2Adam/dense_72/bias/v
&:$ 2Adam/dense_73/kernel/v
 : 2Adam/dense_73/bias/v
&:$  2Adam/dense_74/kernel/v
 : 2Adam/dense_74/bias/v
&:$  2Adam/dense_75/kernel/v
 : 2Adam/dense_75/bias/v
&:$ 2Adam/dense_76/kernel/v
 :2Adam/dense_76/bias/vถ
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021397`
$%,-45<=4ข1
*ข'

input_29
p 

 
ช "ข

0
 ถ
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021427`
$%,-45<=4ข1
*ข'

input_29
p

 
ช "ข

0
 ด
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021550^
$%,-45<=2ข/
(ข%

inputs
p 

 
ช "ข

0
 ด
R__inference_Flatten5Layers_batch8_layer_call_and_return_conditional_losses_1021590^
$%,-45<=2ข/
(ข%

inputs
p

 
ช "ข

0
 
7__inference_Flatten5Layers_batch8_layer_call_fn_1021206S
$%,-45<=4ข1
*ข'

input_29
p 

 
ช "
7__inference_Flatten5Layers_batch8_layer_call_fn_1021367S
$%,-45<=4ข1
*ข'

input_29
p

 
ช "
7__inference_Flatten5Layers_batch8_layer_call_fn_1021485Q
$%,-45<=2ข/
(ข%

inputs
p 

 
ช "
7__inference_Flatten5Layers_batch8_layer_call_fn_1021510Q
$%,-45<=2ข/
(ข%

inputs
p

 
ช "
"__inference__wrapped_model_1021083f
$%,-45<=,ข)
"ข

input_29
ช "*ช'
%
dense_76
dense_76
E__inference_dense_72_layer_call_and_return_conditional_losses_1021621J&ข#
ข

inputs
ช "ข

0
 k
*__inference_dense_72_layer_call_fn_1021610=&ข#
ข

inputs
ช "
E__inference_dense_73_layer_call_and_return_conditional_losses_1021641J$%&ข#
ข

inputs
ช "ข

0 
 k
*__inference_dense_73_layer_call_fn_1021630=$%&ข#
ข

inputs
ช " 
E__inference_dense_74_layer_call_and_return_conditional_losses_1021661J,-&ข#
ข

inputs 
ช "ข

0 
 k
*__inference_dense_74_layer_call_fn_1021650=,-&ข#
ข

inputs 
ช " 
E__inference_dense_75_layer_call_and_return_conditional_losses_1021681J45&ข#
ข

inputs 
ช "ข

0 
 k
*__inference_dense_75_layer_call_fn_1021670=45&ข#
ข

inputs 
ช " 
E__inference_dense_76_layer_call_and_return_conditional_losses_1021700J<=&ข#
ข

inputs 
ช "ข

0
 k
*__inference_dense_76_layer_call_fn_1021690=<=&ข#
ข

inputs 
ช "
G__inference_flatten_20_layer_call_and_return_conditional_losses_1021601J*ข'
 ข

inputs
ช "ข

0
 m
,__inference_flatten_20_layer_call_fn_1021595=*ข'
 ข

inputs
ช "
%__inference_signature_wrapper_1021460r
$%,-45<=8ข5
ข 
.ช+
)
input_29
input_29"*ช'
%
dense_76
dense_76