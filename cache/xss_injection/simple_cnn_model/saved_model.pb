??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
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
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
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
executor_typestring ?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
{
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*
shared_nameconv1d/kernel
t
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*#
_output_shapes
:?@*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:@*
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:@@*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:@*
dtype0

conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?* 
shared_nameconv1d_2/kernel
x
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*#
_output_shapes
:@?*
dtype0
s
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv1d_2/bias
l
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes	
:?*
dtype0
?
conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??* 
shared_nameconv1d_3/kernel
y
#conv1d_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_3/kernel*$
_output_shapes
:??*
dtype0
s
conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv1d_3/bias
l
!conv1d_3/bias/Read/ReadVariableOpReadVariableOpconv1d_3/bias*
_output_shapes	
:?*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?=?*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
?=?*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
w
Output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_nameOutput/kernel
p
!Output/kernel/Read/ReadVariableOpReadVariableOpOutput/kernel*
_output_shapes
:	?*
dtype0
n
Output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameOutput/bias
g
Output/bias/Read/ReadVariableOpReadVariableOpOutput/bias*
_output_shapes
:*
dtype0
x
training/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *#
shared_nametraining/Adam/iter
q
&training/Adam/iter/Read/ReadVariableOpReadVariableOptraining/Adam/iter*
_output_shapes
: *
dtype0	
|
training/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nametraining/Adam/beta_1
u
(training/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining/Adam/beta_1*
_output_shapes
: *
dtype0
|
training/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nametraining/Adam/beta_2
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
_output_shapes
: *
dtype0
z
training/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nametraining/Adam/decay
s
'training/Adam/decay/Read/ReadVariableOpReadVariableOptraining/Adam/decay*
_output_shapes
: *
dtype0
?
training/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nametraining/Adam/learning_rate
?
/training/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
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
?
training/Adam/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*.
shared_nametraining/Adam/conv1d/kernel/m
?
1training/Adam/conv1d/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv1d/kernel/m*#
_output_shapes
:?@*
dtype0
?
training/Adam/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nametraining/Adam/conv1d/bias/m
?
/training/Adam/conv1d/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv1d/bias/m*
_output_shapes
:@*
dtype0
?
training/Adam/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*0
shared_name!training/Adam/conv1d_1/kernel/m
?
3training/Adam/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv1d_1/kernel/m*"
_output_shapes
:@@*
dtype0
?
training/Adam/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nametraining/Adam/conv1d_1/bias/m
?
1training/Adam/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv1d_1/bias/m*
_output_shapes
:@*
dtype0
?
training/Adam/conv1d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*0
shared_name!training/Adam/conv1d_2/kernel/m
?
3training/Adam/conv1d_2/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv1d_2/kernel/m*#
_output_shapes
:@?*
dtype0
?
training/Adam/conv1d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*.
shared_nametraining/Adam/conv1d_2/bias/m
?
1training/Adam/conv1d_2/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv1d_2/bias/m*
_output_shapes	
:?*
dtype0
?
training/Adam/conv1d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*0
shared_name!training/Adam/conv1d_3/kernel/m
?
3training/Adam/conv1d_3/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv1d_3/kernel/m*$
_output_shapes
:??*
dtype0
?
training/Adam/conv1d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*.
shared_nametraining/Adam/conv1d_3/bias/m
?
1training/Adam/conv1d_3/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv1d_3/bias/m*
_output_shapes	
:?*
dtype0
?
training/Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?=?*-
shared_nametraining/Adam/dense/kernel/m
?
0training/Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/m* 
_output_shapes
:
?=?*
dtype0
?
training/Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nametraining/Adam/dense/bias/m
?
.training/Adam/dense/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/m*
_output_shapes	
:?*
dtype0
?
training/Adam/Output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*.
shared_nametraining/Adam/Output/kernel/m
?
1training/Adam/Output/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/Output/kernel/m*
_output_shapes
:	?*
dtype0
?
training/Adam/Output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nametraining/Adam/Output/bias/m
?
/training/Adam/Output/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/Output/bias/m*
_output_shapes
:*
dtype0
?
training/Adam/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*.
shared_nametraining/Adam/conv1d/kernel/v
?
1training/Adam/conv1d/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv1d/kernel/v*#
_output_shapes
:?@*
dtype0
?
training/Adam/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nametraining/Adam/conv1d/bias/v
?
/training/Adam/conv1d/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv1d/bias/v*
_output_shapes
:@*
dtype0
?
training/Adam/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*0
shared_name!training/Adam/conv1d_1/kernel/v
?
3training/Adam/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv1d_1/kernel/v*"
_output_shapes
:@@*
dtype0
?
training/Adam/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nametraining/Adam/conv1d_1/bias/v
?
1training/Adam/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv1d_1/bias/v*
_output_shapes
:@*
dtype0
?
training/Adam/conv1d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*0
shared_name!training/Adam/conv1d_2/kernel/v
?
3training/Adam/conv1d_2/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv1d_2/kernel/v*#
_output_shapes
:@?*
dtype0
?
training/Adam/conv1d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*.
shared_nametraining/Adam/conv1d_2/bias/v
?
1training/Adam/conv1d_2/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv1d_2/bias/v*
_output_shapes	
:?*
dtype0
?
training/Adam/conv1d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*0
shared_name!training/Adam/conv1d_3/kernel/v
?
3training/Adam/conv1d_3/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv1d_3/kernel/v*$
_output_shapes
:??*
dtype0
?
training/Adam/conv1d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*.
shared_nametraining/Adam/conv1d_3/bias/v
?
1training/Adam/conv1d_3/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv1d_3/bias/v*
_output_shapes	
:?*
dtype0
?
training/Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?=?*-
shared_nametraining/Adam/dense/kernel/v
?
0training/Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/v* 
_output_shapes
:
?=?*
dtype0
?
training/Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nametraining/Adam/dense/bias/v
?
.training/Adam/dense/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/v*
_output_shapes	
:?*
dtype0
?
training/Adam/Output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*.
shared_nametraining/Adam/Output/kernel/v
?
1training/Adam/Output/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/Output/kernel/v*
_output_shapes
:	?*
dtype0
?
training/Adam/Output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nametraining/Adam/Output/bias/v
?
/training/Adam/Output/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/Output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?M
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?M
value?MB?M B?M
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
 trainable_variables
!	variables
"	keras_api
R
#regularization_losses
$trainable_variables
%	variables
&	keras_api
h

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
h

-kernel
.bias
/regularization_losses
0trainable_variables
1	variables
2	keras_api
R
3regularization_losses
4trainable_variables
5	variables
6	keras_api
R
7regularization_losses
8trainable_variables
9	variables
:	keras_api
R
;regularization_losses
<trainable_variables
=	variables
>	keras_api
h

?kernel
@bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
R
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
h

Ikernel
Jbias
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
?
Oiter

Pbeta_1

Qbeta_2
	Rdecay
Slearning_ratem?m?m?m?'m?(m?-m?.m??m?@m?Im?Jm?v?v?v?v?'v?(v?-v?.v??v?@v?Iv?Jv?
 
V
0
1
2
3
'4
(5
-6
.7
?8
@9
I10
J11
V
0
1
2
3
'4
(5
-6
.7
?8
@9
I10
J11
?
Tlayer_regularization_losses
Unon_trainable_variables

Vlayers
Wlayer_metrics
regularization_losses
trainable_variables
	variables
Xmetrics
 
YW
VARIABLE_VALUEconv1d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Ylayer_regularization_losses

Zlayers
[non_trainable_variables
\layer_metrics
regularization_losses
trainable_variables
	variables
]metrics
[Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
^layer_regularization_losses

_layers
`non_trainable_variables
alayer_metrics
regularization_losses
trainable_variables
	variables
bmetrics
 
 
 
?
clayer_regularization_losses

dlayers
enon_trainable_variables
flayer_metrics
regularization_losses
 trainable_variables
!	variables
gmetrics
 
 
 
?
hlayer_regularization_losses

ilayers
jnon_trainable_variables
klayer_metrics
#regularization_losses
$trainable_variables
%	variables
lmetrics
[Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
?
mlayer_regularization_losses

nlayers
onon_trainable_variables
player_metrics
)regularization_losses
*trainable_variables
+	variables
qmetrics
[Y
VARIABLE_VALUEconv1d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1

-0
.1
?
rlayer_regularization_losses

slayers
tnon_trainable_variables
ulayer_metrics
/regularization_losses
0trainable_variables
1	variables
vmetrics
 
 
 
?
wlayer_regularization_losses

xlayers
ynon_trainable_variables
zlayer_metrics
3regularization_losses
4trainable_variables
5	variables
{metrics
 
 
 
?
|layer_regularization_losses

}layers
~non_trainable_variables
layer_metrics
7regularization_losses
8trainable_variables
9	variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?layers
?non_trainable_variables
?layer_metrics
;regularization_losses
<trainable_variables
=	variables
?metrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
@1

?0
@1
?
 ?layer_regularization_losses
?layers
?non_trainable_variables
?layer_metrics
Aregularization_losses
Btrainable_variables
C	variables
?metrics
 
 
 
?
 ?layer_regularization_losses
?layers
?non_trainable_variables
?layer_metrics
Eregularization_losses
Ftrainable_variables
G	variables
?metrics
YW
VARIABLE_VALUEOutput/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEOutput/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

I0
J1

I0
J1
?
 ?layer_regularization_losses
?layers
?non_trainable_variables
?layer_metrics
Kregularization_losses
Ltrainable_variables
M	variables
?metrics
QO
VARIABLE_VALUEtraining/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEtraining/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEtraining/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
V
0
1
2
3
4
5
6
7
	8

9
10
11
 

?0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUEtraining/Adam/conv1d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEtraining/Adam/conv1d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv1d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv1d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv1d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv1d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv1d_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv1d_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEtraining/Adam/dense/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/Output/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEtraining/Adam/Output/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv1d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEtraining/Adam/conv1d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv1d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv1d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv1d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv1d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv1d_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv1d_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEtraining/Adam/dense/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/Output/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEtraining/Adam/Output/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
v
serving_default_input_1Placeholder*%
_output_shapes
:???*
dtype0*
shape:???
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biasdense/kernel
dense/biasOutput/kernelOutput/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_3993
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp#conv1d_3/kernel/Read/ReadVariableOp!conv1d_3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp!Output/kernel/Read/ReadVariableOpOutput/bias/Read/ReadVariableOp&training/Adam/iter/Read/ReadVariableOp(training/Adam/beta_1/Read/ReadVariableOp(training/Adam/beta_2/Read/ReadVariableOp'training/Adam/decay/Read/ReadVariableOp/training/Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1training/Adam/conv1d/kernel/m/Read/ReadVariableOp/training/Adam/conv1d/bias/m/Read/ReadVariableOp3training/Adam/conv1d_1/kernel/m/Read/ReadVariableOp1training/Adam/conv1d_1/bias/m/Read/ReadVariableOp3training/Adam/conv1d_2/kernel/m/Read/ReadVariableOp1training/Adam/conv1d_2/bias/m/Read/ReadVariableOp3training/Adam/conv1d_3/kernel/m/Read/ReadVariableOp1training/Adam/conv1d_3/bias/m/Read/ReadVariableOp0training/Adam/dense/kernel/m/Read/ReadVariableOp.training/Adam/dense/bias/m/Read/ReadVariableOp1training/Adam/Output/kernel/m/Read/ReadVariableOp/training/Adam/Output/bias/m/Read/ReadVariableOp1training/Adam/conv1d/kernel/v/Read/ReadVariableOp/training/Adam/conv1d/bias/v/Read/ReadVariableOp3training/Adam/conv1d_1/kernel/v/Read/ReadVariableOp1training/Adam/conv1d_1/bias/v/Read/ReadVariableOp3training/Adam/conv1d_2/kernel/v/Read/ReadVariableOp1training/Adam/conv1d_2/bias/v/Read/ReadVariableOp3training/Adam/conv1d_3/kernel/v/Read/ReadVariableOp1training/Adam/conv1d_3/bias/v/Read/ReadVariableOp0training/Adam/dense/kernel/v/Read/ReadVariableOp.training/Adam/dense/bias/v/Read/ReadVariableOp1training/Adam/Output/kernel/v/Read/ReadVariableOp/training/Adam/Output/bias/v/Read/ReadVariableOpConst*8
Tin1
/2-	*
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
GPU 2J 8? *&
f!R
__inference__traced_save_4578
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biasdense/kernel
dense/biasOutput/kernelOutput/biastraining/Adam/itertraining/Adam/beta_1training/Adam/beta_2training/Adam/decaytraining/Adam/learning_ratetotalcounttraining/Adam/conv1d/kernel/mtraining/Adam/conv1d/bias/mtraining/Adam/conv1d_1/kernel/mtraining/Adam/conv1d_1/bias/mtraining/Adam/conv1d_2/kernel/mtraining/Adam/conv1d_2/bias/mtraining/Adam/conv1d_3/kernel/mtraining/Adam/conv1d_3/bias/mtraining/Adam/dense/kernel/mtraining/Adam/dense/bias/mtraining/Adam/Output/kernel/mtraining/Adam/Output/bias/mtraining/Adam/conv1d/kernel/vtraining/Adam/conv1d/bias/vtraining/Adam/conv1d_1/kernel/vtraining/Adam/conv1d_1/bias/vtraining/Adam/conv1d_2/kernel/vtraining/Adam/conv1d_2/bias/vtraining/Adam/conv1d_3/kernel/vtraining/Adam/conv1d_3/bias/vtraining/Adam/dense/kernel/vtraining/Adam/dense/bias/vtraining/Adam/Output/kernel/vtraining/Adam/Output/bias/v*7
Tin0
.2,*
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
GPU 2J 8? *)
f$R"
 __inference__traced_restore_4717??

?
?
B__inference_conv1d_3_layer_call_and_return_conditional_losses_3716

inputs6
2conv1d_expanddims_1_readvariableop_conv1d_3_kernel(
$biasadd_readvariableop_conv1d_3_bias
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*(
_output_shapes
:?}?2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_expanddims_1_readvariableop_conv1d_3_kernel*$
_output_shapes
:??*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:?{?*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*$
_output_shapes
:?{?*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv1d_3_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:?{?2	
BiasAddU
ReluReluBiasAdd:output:0*
T0*$
_output_shapes
:?{?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*$
_output_shapes
:?{?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?}?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:L H
$
_output_shapes
:?}?
 
_user_specified_nameinputs
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_4374

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel* 
_output_shapes
:
?=?*
dtype02
MatMul/ReadVariableOpl
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpz
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2	
BiasAddQ
ReluReluBiasAdd:output:0*
T0* 
_output_shapes
:
??2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0* 
_output_shapes
:
??2

Identity"
identityIdentity:output:0*'
_input_shapes
:
??=::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:H D
 
_output_shapes
:
??=
 
_user_specified_nameinputs
?
]
A__inference_flatten_layer_call_and_return_conditional_losses_3770

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Const`
ReshapeReshapeinputsConst:output:0*
T0* 
_output_shapes
:
??=2	
Reshape]
IdentityIdentityReshape:output:0*
T0* 
_output_shapes
:
??=2

Identity"
identityIdentity:output:0*#
_input_shapes
:?=?:L H
$
_output_shapes
:?=?
 
_user_specified_nameinputs
?	
?
"__inference_signature_wrapper_3993
input_1
conv1d_kernel
conv1d_bias
conv1d_1_kernel
conv1d_1_bias
conv1d_2_kernel
conv1d_2_bias
conv1d_3_kernel
conv1d_3_bias
dense_kernel

dense_bias
output_kernel
output_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_kernelconv1d_biasconv1d_1_kernelconv1d_1_biasconv1d_2_kernelconv1d_2_biasconv1d_3_kernelconv1d_3_biasdense_kernel
dense_biasoutput_kerneloutput_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_35292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:???::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
%
_output_shapes
:???
!
_user_specified_name	input_1
?
]
A__inference_flatten_layer_call_and_return_conditional_losses_4358

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Const`
ReshapeReshapeinputsConst:output:0*
T0* 
_output_shapes
:
??=2	
Reshape]
IdentityIdentityReshape:output:0*
T0* 
_output_shapes
:
??=2

Identity"
identityIdentity:output:0*#
_input_shapes
:?=?:L H
$
_output_shapes
:?=?
 
_user_specified_nameinputs
?

b
C__inference_dropout_2_layer_call_and_return_conditional_losses_3813

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constl
dropout/MulMulinputsdropout/Const:output:0*
T0* 
_output_shapes
:
??2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?  ?   2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0* 
_output_shapes
:
??*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0* 
_output_shapes
:
??2
dropout/GreaterEqualx
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
* 
_output_shapes
:
??2
dropout/Casts
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0* 
_output_shapes
:
??2
dropout/Mul_1^
IdentityIdentitydropout/Mul_1:z:0*
T0* 
_output_shapes
:
??2

Identity"
identityIdentity:output:0*
_input_shapes
:
??:H D
 
_output_shapes
:
??
 
_user_specified_nameinputs
?

`
A__inference_dropout_layer_call_and_return_conditional_losses_3654

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consto
dropout/MulMulinputsdropout/Const:output:0*
T0*#
_output_shapes
:?@2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"?     @   2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*#
_output_shapes
:?@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?@2
dropout/GreaterEqual{
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?@2
dropout/Castv
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*#
_output_shapes
:?@2
dropout/Mul_1a
IdentityIdentitydropout/Mul_1:z:0*
T0*#
_output_shapes
:?@2

Identity"
identityIdentity:output:0*"
_input_shapes
:?@:K G
#
_output_shapes
:?@
 
_user_specified_nameinputs
?

$__inference_dense_layer_call_fn_4381

inputs
dense_kernel

dense_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
??*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_37892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0* 
_output_shapes
:
??2

Identity"
identityIdentity:output:0*'
_input_shapes
:
??=::22
StatefulPartitionedCallStatefulPartitionedCall:H D
 
_output_shapes
:
??=
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_3659

inputs

identity_1V
IdentityIdentityinputs*
T0*#
_output_shapes
:?@2

Identitye

Identity_1IdentityIdentity:output:0*
T0*#
_output_shapes
:?@2

Identity_1"!

identity_1Identity_1:output:0*"
_input_shapes
:?@:K G
#
_output_shapes
:?@
 
_user_specified_nameinputs
?1
?
D__inference_sequential_layer_call_and_return_conditional_losses_3959

inputs
conv1d_conv1d_kernel
conv1d_conv1d_bias
conv1d_1_conv1d_1_kernel
conv1d_1_conv1d_1_bias
conv1d_2_conv1d_2_kernel
conv1d_2_conv1d_2_bias
conv1d_3_conv1d_3_kernel
conv1d_3_conv1d_3_bias
dense_dense_kernel
dense_dense_bias
output_output_kernel
output_output_bias
identity??Output/StatefulPartitionedCall?conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_conv1d_kernelconv1d_conv1d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:??@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_35952 
conv1d/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_conv1d_1_kernelconv1d_1_conv1d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:??@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_36232"
 conv1d_1/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_35492
max_pooling1d/PartitionedCall?
dropout/PartitionedCallPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_36592
dropout/PartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv1d_2_conv1d_2_kernelconv1d_2_conv1d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?}?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_36882"
 conv1d_2/StatefulPartitionedCall?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0conv1d_3_conv1d_3_kernelconv1d_3_conv1d_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?{?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_3_layer_call_and_return_conditional_losses_37162"
 conv1d_3/StatefulPartitionedCall?
max_pooling1d_1/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?=?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_35722!
max_pooling1d_1/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?=?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_37512
dropout_1/PartitionedCall?
flatten/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
??=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_37702
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
??*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_37892
dense/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
??* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_38182
dropout_2/PartitionedCall?
Output/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0output_output_kerneloutput_output_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Output_layer_call_and_return_conditional_losses_38422 
Output/StatefulPartitionedCall?
IdentityIdentity'Output/StatefulPartitionedCall:output:0^Output/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:???::::::::::::2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:M I
%
_output_shapes
:???
 
_user_specified_nameinputs
?
?
@__inference_conv1d_layer_call_and_return_conditional_losses_4222

inputs4
0conv1d_expanddims_1_readvariableop_conv1d_kernel&
"biasadd_readvariableop_conv1d_bias
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*)
_output_shapes
:???2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0conv1d_expanddims_1_readvariableop_conv1d_kernel*#
_output_shapes
:?@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:??@*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*$
_output_shapes
:??@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_conv1d_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:??@2	
BiasAddU
ReluReluBiasAdd:output:0*
T0*$
_output_shapes
:??@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*$
_output_shapes
:??@2

Identity"
identityIdentity:output:0*,
_input_shapes
:???::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:M I
%
_output_shapes
:???
 
_user_specified_nameinputs
?g
?	
D__inference_sequential_layer_call_and_return_conditional_losses_4172

inputs;
7conv1d_conv1d_expanddims_1_readvariableop_conv1d_kernel-
)conv1d_biasadd_readvariableop_conv1d_bias?
;conv1d_1_conv1d_expanddims_1_readvariableop_conv1d_1_kernel1
-conv1d_1_biasadd_readvariableop_conv1d_1_bias?
;conv1d_2_conv1d_expanddims_1_readvariableop_conv1d_2_kernel1
-conv1d_2_biasadd_readvariableop_conv1d_2_bias?
;conv1d_3_conv1d_expanddims_1_readvariableop_conv1d_3_kernel1
-conv1d_3_biasadd_readvariableop_conv1d_3_bias,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias.
*output_matmul_readvariableop_output_kernel-
)output_biasadd_readvariableop_output_bias
identity??Output/BiasAdd/ReadVariableOp?Output/MatMul/ReadVariableOp?conv1d/BiasAdd/ReadVariableOp?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?conv1d_1/BiasAdd/ReadVariableOp?+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?conv1d_2/BiasAdd/ReadVariableOp?+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?conv1d_3/BiasAdd/ReadVariableOp?+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDimsinputs%conv1d/conv1d/ExpandDims/dim:output:0*
T0*)
_output_shapes
:???2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp7conv1d_conv1d_expanddims_1_readvariableop_conv1d_kernel*#
_output_shapes
:?@*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?@2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:??@*
paddingVALID*
strides
2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*$
_output_shapes
:??@*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp)conv1d_biasadd_readvariableop_conv1d_bias*
_output_shapes
:@*
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:??@2
conv1d/BiasAddj
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*$
_output_shapes
:??@2
conv1d/Relu?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*(
_output_shapes
:??@2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;conv1d_1_conv1d_expanddims_1_readvariableop_conv1d_1_kernel*"
_output_shapes
:@@*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:??@*
paddingVALID*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*$
_output_shapes
:??@*
squeeze_dims

?????????2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp-conv1d_1_biasadd_readvariableop_conv1d_1_bias*
_output_shapes
:@*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:??@2
conv1d_1/BiasAddp
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*$
_output_shapes
:??@2
conv1d_1/Relu~
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
max_pooling1d/ExpandDims/dim?
max_pooling1d/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0%max_pooling1d/ExpandDims/dim:output:0*
T0*(
_output_shapes
:??@2
max_pooling1d/ExpandDims?
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*'
_output_shapes
:?@*
ksize
*
paddingVALID*
strides
2
max_pooling1d/MaxPool?
max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*#
_output_shapes
:?@*
squeeze_dims
2
max_pooling1d/Squeeze~
dropout/IdentityIdentitymax_pooling1d/Squeeze:output:0*
T0*#
_output_shapes
:?@2
dropout/Identity?
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_2/conv1d/ExpandDims/dim?
conv1d_2/conv1d/ExpandDims
ExpandDimsdropout/Identity:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?@2
conv1d_2/conv1d/ExpandDims?
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;conv1d_2_conv1d_expanddims_1_readvariableop_conv1d_2_kernel*#
_output_shapes
:@?*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:?}?*
paddingVALID*
strides
2
conv1d_2/conv1d?
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*$
_output_shapes
:?}?*
squeeze_dims

?????????2
conv1d_2/conv1d/Squeeze?
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp-conv1d_2_biasadd_readvariableop_conv1d_2_bias*
_output_shapes	
:?*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp?
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:?}?2
conv1d_2/BiasAddp
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*$
_output_shapes
:?}?2
conv1d_2/Relu?
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_3/conv1d/ExpandDims/dim?
conv1d_3/conv1d/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*(
_output_shapes
:?}?2
conv1d_3/conv1d/ExpandDims?
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;conv1d_3_conv1d_expanddims_1_readvariableop_conv1d_3_kernel*$
_output_shapes
:??*
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dim?
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_3/conv1d/ExpandDims_1?
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:?{?*
paddingVALID*
strides
2
conv1d_3/conv1d?
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*$
_output_shapes
:?{?*
squeeze_dims

?????????2
conv1d_3/conv1d/Squeeze?
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp-conv1d_3_biasadd_readvariableop_conv1d_3_bias*
_output_shapes	
:?*
dtype02!
conv1d_3/BiasAdd/ReadVariableOp?
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:?{?2
conv1d_3/BiasAddp
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*$
_output_shapes
:?{?2
conv1d_3/Relu?
max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_1/ExpandDims/dim?
max_pooling1d_1/ExpandDims
ExpandDimsconv1d_3/Relu:activations:0'max_pooling1d_1/ExpandDims/dim:output:0*
T0*(
_output_shapes
:?{?2
max_pooling1d_1/ExpandDims?
max_pooling1d_1/MaxPoolMaxPool#max_pooling1d_1/ExpandDims:output:0*(
_output_shapes
:?=?*
ksize
*
paddingVALID*
strides
2
max_pooling1d_1/MaxPool?
max_pooling1d_1/SqueezeSqueeze max_pooling1d_1/MaxPool:output:0*
T0*$
_output_shapes
:?=?*
squeeze_dims
2
max_pooling1d_1/Squeeze?
dropout_1/IdentityIdentity max_pooling1d_1/Squeeze:output:0*
T0*$
_output_shapes
:?=?2
dropout_1/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapedropout_1/Identity:output:0flatten/Const:output:0*
T0* 
_output_shapes
:
??=2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
?=?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
dense/BiasAddc

dense/ReluReludense/BiasAdd:output:0*
T0* 
_output_shapes
:
??2

dense/Reluy
dropout_2/IdentityIdentitydense/Relu:activations:0*
T0* 
_output_shapes
:
??2
dropout_2/Identity?
Output/MatMul/ReadVariableOpReadVariableOp*output_matmul_readvariableop_output_kernel*
_output_shapes
:	?*
dtype02
Output/MatMul/ReadVariableOp?
Output/MatMulMatMuldropout_2/Identity:output:0$Output/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
Output/MatMul?
Output/BiasAdd/ReadVariableOpReadVariableOp)output_biasadd_readvariableop_output_bias*
_output_shapes
:*
dtype02
Output/BiasAdd/ReadVariableOp?
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
Output/BiasAddn
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*
_output_shapes
:	?2
Output/Softmax?
IdentityIdentityOutput/Softmax:softmax:0^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/conv1d/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:???::::::::::::2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:M I
%
_output_shapes
:???
 
_user_specified_nameinputs
?
J
.__inference_max_pooling1d_1_layer_call_fn_3575

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_35722
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_4398

inputs

identity_1S
IdentityIdentityinputs*
T0* 
_output_shapes
:
??2

Identityb

Identity_1IdentityIdentity:output:0*
T0* 
_output_shapes
:
??2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:
??:H D
 
_output_shapes
:
??
 
_user_specified_nameinputs
?
B
&__inference_flatten_layer_call_fn_4363

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
??=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_37702
PartitionedCalle
IdentityIdentityPartitionedCall:output:0*
T0* 
_output_shapes
:
??=2

Identity"
identityIdentity:output:0*#
_input_shapes
:?=?:L H
$
_output_shapes
:?=?
 
_user_specified_nameinputs
?
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_3746

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constp
dropout/MulMulinputsdropout/Const:output:0*
T0*$
_output_shapes
:?=?2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"?  =   ?   2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*$
_output_shapes
:?=?*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*$
_output_shapes
:?=?2
dropout/GreaterEqual|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*$
_output_shapes
:?=?2
dropout/Castw
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*$
_output_shapes
:?=?2
dropout/Mul_1b
IdentityIdentitydropout/Mul_1:z:0*
T0*$
_output_shapes
:?=?2

Identity"
identityIdentity:output:0*#
_input_shapes
:?=?:L H
$
_output_shapes
:?=?
 
_user_specified_nameinputs
?
?
'__inference_conv1d_1_layer_call_fn_4252

inputs
conv1d_1_kernel
conv1d_1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_1_kernelconv1d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:??@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_36232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*$
_output_shapes
:??@2

Identity"
identityIdentity:output:0*+
_input_shapes
:??@::22
StatefulPartitionedCallStatefulPartitionedCall:L H
$
_output_shapes
:??@
 
_user_specified_nameinputs
?
D
(__inference_dropout_2_layer_call_fn_4408

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
??* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_38182
PartitionedCalle
IdentityIdentityPartitionedCall:output:0*
T0* 
_output_shapes
:
??2

Identity"
identityIdentity:output:0*
_input_shapes
:
??:H D
 
_output_shapes
:
??
 
_user_specified_nameinputs
?	
?
)__inference_sequential_layer_call_fn_4189

inputs
conv1d_kernel
conv1d_bias
conv1d_1_kernel
conv1d_1_bias
conv1d_2_kernel
conv1d_2_bias
conv1d_3_kernel
conv1d_3_bias
dense_kernel

dense_bias
output_kernel
output_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_kernelconv1d_biasconv1d_1_kernelconv1d_1_biasconv1d_2_kernelconv1d_2_biasconv1d_3_kernelconv1d_3_biasdense_kernel
dense_biasoutput_kerneloutput_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_39142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:???::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
%
_output_shapes
:???
 
_user_specified_nameinputs
?
?
B__inference_conv1d_2_layer_call_and_return_conditional_losses_4295

inputs6
2conv1d_expanddims_1_readvariableop_conv1d_2_kernel(
$biasadd_readvariableop_conv1d_2_bias
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_expanddims_1_readvariableop_conv1d_2_kernel*#
_output_shapes
:@?*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:?}?*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*$
_output_shapes
:?}?*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv1d_2_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:?}?2	
BiasAddU
ReluReluBiasAdd:output:0*
T0*$
_output_shapes
:?}?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*$
_output_shapes
:?}?2

Identity"
identityIdentity:output:0**
_input_shapes
:?@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:K G
#
_output_shapes
:?@
 
_user_specified_nameinputs
?
_
&__inference_dropout_layer_call_fn_4274

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_36542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:?@2

Identity"
identityIdentity:output:0*"
_input_shapes
:?@22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?@
 
_user_specified_nameinputs
?
?
B__inference_conv1d_2_layer_call_and_return_conditional_losses_3688

inputs6
2conv1d_expanddims_1_readvariableop_conv1d_2_kernel(
$biasadd_readvariableop_conv1d_2_bias
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_expanddims_1_readvariableop_conv1d_2_kernel*#
_output_shapes
:@?*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:?}?*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*$
_output_shapes
:?}?*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv1d_2_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:?}?2	
BiasAddU
ReluReluBiasAdd:output:0*
T0*$
_output_shapes
:?}?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*$
_output_shapes
:?}?2

Identity"
identityIdentity:output:0**
_input_shapes
:?@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:K G
#
_output_shapes
:?@
 
_user_specified_nameinputs
?
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_4337

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constp
dropout/MulMulinputsdropout/Const:output:0*
T0*$
_output_shapes
:?=?2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"?  =   ?   2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*$
_output_shapes
:?=?*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*$
_output_shapes
:?=?2
dropout/GreaterEqual|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*$
_output_shapes
:?=?2
dropout/Castw
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*$
_output_shapes
:?=?2
dropout/Mul_1b
IdentityIdentitydropout/Mul_1:z:0*
T0*$
_output_shapes
:?=?2

Identity"
identityIdentity:output:0*#
_input_shapes
:?=?:L H
$
_output_shapes
:?=?
 
_user_specified_nameinputs
?
H
,__inference_max_pooling1d_layer_call_fn_3552

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_35492
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_4269

inputs

identity_1V
IdentityIdentityinputs*
T0*#
_output_shapes
:?@2

Identitye

Identity_1IdentityIdentity:output:0*
T0*#
_output_shapes
:?@2

Identity_1"!

identity_1Identity_1:output:0*"
_input_shapes
:?@:K G
#
_output_shapes
:?@
 
_user_specified_nameinputs
?6
?
D__inference_sequential_layer_call_and_return_conditional_losses_3914

inputs
conv1d_conv1d_kernel
conv1d_conv1d_bias
conv1d_1_conv1d_1_kernel
conv1d_1_conv1d_1_bias
conv1d_2_conv1d_2_kernel
conv1d_2_conv1d_2_bias
conv1d_3_conv1d_3_kernel
conv1d_3_conv1d_3_bias
dense_dense_kernel
dense_dense_bias
output_output_kernel
output_output_bias
identity??Output/StatefulPartitionedCall?conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_conv1d_kernelconv1d_conv1d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:??@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_35952 
conv1d/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_conv1d_1_kernelconv1d_1_conv1d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:??@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_36232"
 conv1d_1/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_35492
max_pooling1d/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_36542!
dropout/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv1d_2_conv1d_2_kernelconv1d_2_conv1d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?}?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_36882"
 conv1d_2/StatefulPartitionedCall?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0conv1d_3_conv1d_3_kernelconv1d_3_conv1d_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?{?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_3_layer_call_and_return_conditional_losses_37162"
 conv1d_3/StatefulPartitionedCall?
max_pooling1d_1/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?=?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_35722!
max_pooling1d_1/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?=?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_37462#
!dropout_1/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
??=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_37702
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
??*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_37892
dense/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
??* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_38132#
!dropout_2/StatefulPartitionedCall?
Output/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0output_output_kerneloutput_output_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Output_layer_call_and_return_conditional_losses_38422 
Output/StatefulPartitionedCall?
IdentityIdentity'Output/StatefulPartitionedCall:output:0^Output/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:???::::::::::::2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:M I
%
_output_shapes
:???
 
_user_specified_nameinputs
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_3751

inputs

identity_1W
IdentityIdentityinputs*
T0*$
_output_shapes
:?=?2

Identityf

Identity_1IdentityIdentity:output:0*
T0*$
_output_shapes
:?=?2

Identity_1"!

identity_1Identity_1:output:0*#
_input_shapes
:?=?:L H
$
_output_shapes
:?=?
 
_user_specified_nameinputs
?
?
'__inference_conv1d_3_layer_call_fn_4325

inputs
conv1d_3_kernel
conv1d_3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_3_kernelconv1d_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?{?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_3_layer_call_and_return_conditional_losses_37162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*$
_output_shapes
:?{?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?}?::22
StatefulPartitionedCallStatefulPartitionedCall:L H
$
_output_shapes
:?}?
 
_user_specified_nameinputs
?
a
(__inference_dropout_2_layer_call_fn_4403

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
??* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_38132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0* 
_output_shapes
:
??2

Identity"
identityIdentity:output:0*
_input_shapes
:
??22
StatefulPartitionedCallStatefulPartitionedCall:H D
 
_output_shapes
:
??
 
_user_specified_nameinputs
?	
?
)__inference_sequential_layer_call_fn_3929
input_1
conv1d_kernel
conv1d_bias
conv1d_1_kernel
conv1d_1_bias
conv1d_2_kernel
conv1d_2_bias
conv1d_3_kernel
conv1d_3_bias
dense_kernel

dense_bias
output_kernel
output_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_kernelconv1d_biasconv1d_1_kernelconv1d_1_biasconv1d_2_kernelconv1d_2_biasconv1d_3_kernelconv1d_3_biasdense_kernel
dense_biasoutput_kerneloutput_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_39142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:???::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
%
_output_shapes
:???
!
_user_specified_name	input_1
?
?
'__inference_conv1d_2_layer_call_fn_4302

inputs
conv1d_2_kernel
conv1d_2_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_2_kernelconv1d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?}?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_36882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*$
_output_shapes
:?}?2

Identity"
identityIdentity:output:0**
_input_shapes
:?@::22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?@
 
_user_specified_nameinputs
?
a
(__inference_dropout_1_layer_call_fn_4347

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?=?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_37462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*$
_output_shapes
:?=?2

Identity"
identityIdentity:output:0*#
_input_shapes
:?=?22
StatefulPartitionedCallStatefulPartitionedCall:L H
$
_output_shapes
:?=?
 
_user_specified_nameinputs
?
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_3818

inputs

identity_1S
IdentityIdentityinputs*
T0* 
_output_shapes
:
??2

Identityb

Identity_1IdentityIdentity:output:0*
T0* 
_output_shapes
:
??2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:
??:H D
 
_output_shapes
:
??
 
_user_specified_nameinputs
?
?
%__inference_conv1d_layer_call_fn_4229

inputs
conv1d_kernel
conv1d_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_kernelconv1d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:??@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_35952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*$
_output_shapes
:??@2

Identity"
identityIdentity:output:0*,
_input_shapes
:???::22
StatefulPartitionedCallStatefulPartitionedCall:M I
%
_output_shapes
:???
 
_user_specified_nameinputs
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_3789

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel* 
_output_shapes
:
?=?*
dtype02
MatMul/ReadVariableOpl
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpz
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2	
BiasAddQ
ReluReluBiasAdd:output:0*
T0* 
_output_shapes
:
??2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0* 
_output_shapes
:
??2

Identity"
identityIdentity:output:0*'
_input_shapes
:
??=::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:H D
 
_output_shapes
:
??=
 
_user_specified_nameinputs
?
c
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_3538

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
%__inference_Output_layer_call_fn_4426

inputs
output_kernel
output_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsoutput_kerneloutput_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Output_layer_call_and_return_conditional_losses_38422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*'
_input_shapes
:
??::22
StatefulPartitionedCallStatefulPartitionedCall:H D
 
_output_shapes
:
??
 
_user_specified_nameinputs
?1
?
D__inference_sequential_layer_call_and_return_conditional_losses_3883
input_1
conv1d_conv1d_kernel
conv1d_conv1d_bias
conv1d_1_conv1d_1_kernel
conv1d_1_conv1d_1_bias
conv1d_2_conv1d_2_kernel
conv1d_2_conv1d_2_bias
conv1d_3_conv1d_3_kernel
conv1d_3_conv1d_3_bias
dense_dense_kernel
dense_dense_bias
output_output_kernel
output_output_bias
identity??Output/StatefulPartitionedCall?conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_conv1d_kernelconv1d_conv1d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:??@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_35952 
conv1d/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_conv1d_1_kernelconv1d_1_conv1d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:??@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_36232"
 conv1d_1/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_35492
max_pooling1d/PartitionedCall?
dropout/PartitionedCallPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_36592
dropout/PartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv1d_2_conv1d_2_kernelconv1d_2_conv1d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?}?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_36882"
 conv1d_2/StatefulPartitionedCall?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0conv1d_3_conv1d_3_kernelconv1d_3_conv1d_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?{?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_3_layer_call_and_return_conditional_losses_37162"
 conv1d_3/StatefulPartitionedCall?
max_pooling1d_1/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?=?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_35722!
max_pooling1d_1/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?=?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_37512
dropout_1/PartitionedCall?
flatten/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
??=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_37702
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
??*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_37892
dense/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
??* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_38182
dropout_2/PartitionedCall?
Output/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0output_output_kerneloutput_output_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Output_layer_call_and_return_conditional_losses_38422 
Output/StatefulPartitionedCall?
IdentityIdentity'Output/StatefulPartitionedCall:output:0^Output/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:???::::::::::::2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:N J
%
_output_shapes
:???
!
_user_specified_name	input_1
?
B
&__inference_dropout_layer_call_fn_4279

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_36592
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0*#
_output_shapes
:?@2

Identity"
identityIdentity:output:0*"
_input_shapes
:?@:K G
#
_output_shapes
:?@
 
_user_specified_nameinputs
?	
?
)__inference_sequential_layer_call_fn_4206

inputs
conv1d_kernel
conv1d_bias
conv1d_1_kernel
conv1d_1_bias
conv1d_2_kernel
conv1d_2_bias
conv1d_3_kernel
conv1d_3_bias
dense_kernel

dense_bias
output_kernel
output_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_kernelconv1d_biasconv1d_1_kernelconv1d_1_biasconv1d_2_kernelconv1d_2_biasconv1d_3_kernelconv1d_3_biasdense_kernel
dense_biasoutput_kerneloutput_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_39592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:???::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
%
_output_shapes
:???
 
_user_specified_nameinputs
?6
?
D__inference_sequential_layer_call_and_return_conditional_losses_3855
input_1
conv1d_conv1d_kernel
conv1d_conv1d_bias
conv1d_1_conv1d_1_kernel
conv1d_1_conv1d_1_bias
conv1d_2_conv1d_2_kernel
conv1d_2_conv1d_2_bias
conv1d_3_conv1d_3_kernel
conv1d_3_conv1d_3_bias
dense_dense_kernel
dense_dense_bias
output_output_kernel
output_output_bias
identity??Output/StatefulPartitionedCall?conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_conv1d_kernelconv1d_conv1d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:??@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_35952 
conv1d/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_conv1d_1_kernelconv1d_1_conv1d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:??@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_36232"
 conv1d_1/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_35492
max_pooling1d/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_36542!
dropout/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv1d_2_conv1d_2_kernelconv1d_2_conv1d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?}?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_36882"
 conv1d_2/StatefulPartitionedCall?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0conv1d_3_conv1d_3_kernelconv1d_3_conv1d_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?{?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_3_layer_call_and_return_conditional_losses_37162"
 conv1d_3/StatefulPartitionedCall?
max_pooling1d_1/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?=?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_35722!
max_pooling1d_1/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?=?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_37462#
!dropout_1/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
??=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_37702
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
??*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_37892
dense/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
??* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_38132#
!dropout_2/StatefulPartitionedCall?
Output/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0output_output_kerneloutput_output_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Output_layer_call_and_return_conditional_losses_38422 
Output/StatefulPartitionedCall?
IdentityIdentity'Output/StatefulPartitionedCall:output:0^Output/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:???::::::::::::2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:N J
%
_output_shapes
:???
!
_user_specified_name	input_1
?	
?
@__inference_Output_layer_call_and_return_conditional_losses_3842

inputs'
#matmul_readvariableop_output_kernel&
"biasadd_readvariableop_output_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp#matmul_readvariableop_output_kernel*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_output_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2	
BiasAddY
SoftmaxSoftmaxBiasAdd:output:0*
T0*
_output_shapes
:	?2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*'
_input_shapes
:
??::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:H D
 
_output_shapes
:
??
 
_user_specified_nameinputs
?
c
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_3549

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_conv1d_1_layer_call_and_return_conditional_losses_4245

inputs6
2conv1d_expanddims_1_readvariableop_conv1d_1_kernel(
$biasadd_readvariableop_conv1d_1_bias
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*(
_output_shapes
:??@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_expanddims_1_readvariableop_conv1d_1_kernel*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:??@*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*$
_output_shapes
:??@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv1d_1_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:??@2	
BiasAddU
ReluReluBiasAdd:output:0*
T0*$
_output_shapes
:??@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*$
_output_shapes
:??@2

Identity"
identityIdentity:output:0*+
_input_shapes
:??@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:L H
$
_output_shapes
:??@
 
_user_specified_nameinputs
?
D
(__inference_dropout_1_layer_call_fn_4352

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:?=?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_37512
PartitionedCalli
IdentityIdentityPartitionedCall:output:0*
T0*$
_output_shapes
:?=?2

Identity"
identityIdentity:output:0*#
_input_shapes
:?=?:L H
$
_output_shapes
:?=?
 
_user_specified_nameinputs
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_4342

inputs

identity_1W
IdentityIdentityinputs*
T0*$
_output_shapes
:?=?2

Identityf

Identity_1IdentityIdentity:output:0*
T0*$
_output_shapes
:?=?2

Identity_1"!

identity_1Identity_1:output:0*#
_input_shapes
:?=?:L H
$
_output_shapes
:?=?
 
_user_specified_nameinputs
?}
?
__inference__wrapped_model_3529
input_1F
Bsequential_conv1d_conv1d_expanddims_1_readvariableop_conv1d_kernel8
4sequential_conv1d_biasadd_readvariableop_conv1d_biasJ
Fsequential_conv1d_1_conv1d_expanddims_1_readvariableop_conv1d_1_kernel<
8sequential_conv1d_1_biasadd_readvariableop_conv1d_1_biasJ
Fsequential_conv1d_2_conv1d_expanddims_1_readvariableop_conv1d_2_kernel<
8sequential_conv1d_2_biasadd_readvariableop_conv1d_2_biasJ
Fsequential_conv1d_3_conv1d_expanddims_1_readvariableop_conv1d_3_kernel<
8sequential_conv1d_3_biasadd_readvariableop_conv1d_3_bias7
3sequential_dense_matmul_readvariableop_dense_kernel6
2sequential_dense_biasadd_readvariableop_dense_bias9
5sequential_output_matmul_readvariableop_output_kernel8
4sequential_output_biasadd_readvariableop_output_bias
identity??(sequential/Output/BiasAdd/ReadVariableOp?'sequential/Output/MatMul/ReadVariableOp?(sequential/conv1d/BiasAdd/ReadVariableOp?4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp?*sequential/conv1d_1/BiasAdd/ReadVariableOp?6sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?*sequential/conv1d_2/BiasAdd/ReadVariableOp?6sequential/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?*sequential/conv1d_3/BiasAdd/ReadVariableOp?6sequential/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?
'sequential/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'sequential/conv1d/conv1d/ExpandDims/dim?
#sequential/conv1d/conv1d/ExpandDims
ExpandDimsinput_10sequential/conv1d/conv1d/ExpandDims/dim:output:0*
T0*)
_output_shapes
:???2%
#sequential/conv1d/conv1d/ExpandDims?
4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpBsequential_conv1d_conv1d_expanddims_1_readvariableop_conv1d_kernel*#
_output_shapes
:?@*
dtype026
4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp?
)sequential/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential/conv1d/conv1d/ExpandDims_1/dim?
%sequential/conv1d/conv1d/ExpandDims_1
ExpandDims<sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:02sequential/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?@2'
%sequential/conv1d/conv1d/ExpandDims_1?
sequential/conv1d/conv1dConv2D,sequential/conv1d/conv1d/ExpandDims:output:0.sequential/conv1d/conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:??@*
paddingVALID*
strides
2
sequential/conv1d/conv1d?
 sequential/conv1d/conv1d/SqueezeSqueeze!sequential/conv1d/conv1d:output:0*
T0*$
_output_shapes
:??@*
squeeze_dims

?????????2"
 sequential/conv1d/conv1d/Squeeze?
(sequential/conv1d/BiasAdd/ReadVariableOpReadVariableOp4sequential_conv1d_biasadd_readvariableop_conv1d_bias*
_output_shapes
:@*
dtype02*
(sequential/conv1d/BiasAdd/ReadVariableOp?
sequential/conv1d/BiasAddBiasAdd)sequential/conv1d/conv1d/Squeeze:output:00sequential/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:??@2
sequential/conv1d/BiasAdd?
sequential/conv1d/ReluRelu"sequential/conv1d/BiasAdd:output:0*
T0*$
_output_shapes
:??@2
sequential/conv1d/Relu?
)sequential/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)sequential/conv1d_1/conv1d/ExpandDims/dim?
%sequential/conv1d_1/conv1d/ExpandDims
ExpandDims$sequential/conv1d/Relu:activations:02sequential/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*(
_output_shapes
:??@2'
%sequential/conv1d_1/conv1d/ExpandDims?
6sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpFsequential_conv1d_1_conv1d_expanddims_1_readvariableop_conv1d_1_kernel*"
_output_shapes
:@@*
dtype028
6sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
+sequential/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/conv1d_1/conv1d/ExpandDims_1/dim?
'sequential/conv1d_1/conv1d/ExpandDims_1
ExpandDims>sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:04sequential/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2)
'sequential/conv1d_1/conv1d/ExpandDims_1?
sequential/conv1d_1/conv1dConv2D.sequential/conv1d_1/conv1d/ExpandDims:output:00sequential/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:??@*
paddingVALID*
strides
2
sequential/conv1d_1/conv1d?
"sequential/conv1d_1/conv1d/SqueezeSqueeze#sequential/conv1d_1/conv1d:output:0*
T0*$
_output_shapes
:??@*
squeeze_dims

?????????2$
"sequential/conv1d_1/conv1d/Squeeze?
*sequential/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp8sequential_conv1d_1_biasadd_readvariableop_conv1d_1_bias*
_output_shapes
:@*
dtype02,
*sequential/conv1d_1/BiasAdd/ReadVariableOp?
sequential/conv1d_1/BiasAddBiasAdd+sequential/conv1d_1/conv1d/Squeeze:output:02sequential/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:??@2
sequential/conv1d_1/BiasAdd?
sequential/conv1d_1/ReluRelu$sequential/conv1d_1/BiasAdd:output:0*
T0*$
_output_shapes
:??@2
sequential/conv1d_1/Relu?
'sequential/max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential/max_pooling1d/ExpandDims/dim?
#sequential/max_pooling1d/ExpandDims
ExpandDims&sequential/conv1d_1/Relu:activations:00sequential/max_pooling1d/ExpandDims/dim:output:0*
T0*(
_output_shapes
:??@2%
#sequential/max_pooling1d/ExpandDims?
 sequential/max_pooling1d/MaxPoolMaxPool,sequential/max_pooling1d/ExpandDims:output:0*'
_output_shapes
:?@*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling1d/MaxPool?
 sequential/max_pooling1d/SqueezeSqueeze)sequential/max_pooling1d/MaxPool:output:0*
T0*#
_output_shapes
:?@*
squeeze_dims
2"
 sequential/max_pooling1d/Squeeze?
sequential/dropout/IdentityIdentity)sequential/max_pooling1d/Squeeze:output:0*
T0*#
_output_shapes
:?@2
sequential/dropout/Identity?
)sequential/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)sequential/conv1d_2/conv1d/ExpandDims/dim?
%sequential/conv1d_2/conv1d/ExpandDims
ExpandDims$sequential/dropout/Identity:output:02sequential/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?@2'
%sequential/conv1d_2/conv1d/ExpandDims?
6sequential/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpFsequential_conv1d_2_conv1d_expanddims_1_readvariableop_conv1d_2_kernel*#
_output_shapes
:@?*
dtype028
6sequential/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
+sequential/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/conv1d_2/conv1d/ExpandDims_1/dim?
'sequential/conv1d_2/conv1d/ExpandDims_1
ExpandDims>sequential/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:04sequential/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2)
'sequential/conv1d_2/conv1d/ExpandDims_1?
sequential/conv1d_2/conv1dConv2D.sequential/conv1d_2/conv1d/ExpandDims:output:00sequential/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:?}?*
paddingVALID*
strides
2
sequential/conv1d_2/conv1d?
"sequential/conv1d_2/conv1d/SqueezeSqueeze#sequential/conv1d_2/conv1d:output:0*
T0*$
_output_shapes
:?}?*
squeeze_dims

?????????2$
"sequential/conv1d_2/conv1d/Squeeze?
*sequential/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp8sequential_conv1d_2_biasadd_readvariableop_conv1d_2_bias*
_output_shapes	
:?*
dtype02,
*sequential/conv1d_2/BiasAdd/ReadVariableOp?
sequential/conv1d_2/BiasAddBiasAdd+sequential/conv1d_2/conv1d/Squeeze:output:02sequential/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:?}?2
sequential/conv1d_2/BiasAdd?
sequential/conv1d_2/ReluRelu$sequential/conv1d_2/BiasAdd:output:0*
T0*$
_output_shapes
:?}?2
sequential/conv1d_2/Relu?
)sequential/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)sequential/conv1d_3/conv1d/ExpandDims/dim?
%sequential/conv1d_3/conv1d/ExpandDims
ExpandDims&sequential/conv1d_2/Relu:activations:02sequential/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*(
_output_shapes
:?}?2'
%sequential/conv1d_3/conv1d/ExpandDims?
6sequential/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpFsequential_conv1d_3_conv1d_expanddims_1_readvariableop_conv1d_3_kernel*$
_output_shapes
:??*
dtype028
6sequential/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?
+sequential/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/conv1d_3/conv1d/ExpandDims_1/dim?
'sequential/conv1d_3/conv1d/ExpandDims_1
ExpandDims>sequential/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:04sequential/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2)
'sequential/conv1d_3/conv1d/ExpandDims_1?
sequential/conv1d_3/conv1dConv2D.sequential/conv1d_3/conv1d/ExpandDims:output:00sequential/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:?{?*
paddingVALID*
strides
2
sequential/conv1d_3/conv1d?
"sequential/conv1d_3/conv1d/SqueezeSqueeze#sequential/conv1d_3/conv1d:output:0*
T0*$
_output_shapes
:?{?*
squeeze_dims

?????????2$
"sequential/conv1d_3/conv1d/Squeeze?
*sequential/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp8sequential_conv1d_3_biasadd_readvariableop_conv1d_3_bias*
_output_shapes	
:?*
dtype02,
*sequential/conv1d_3/BiasAdd/ReadVariableOp?
sequential/conv1d_3/BiasAddBiasAdd+sequential/conv1d_3/conv1d/Squeeze:output:02sequential/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:?{?2
sequential/conv1d_3/BiasAdd?
sequential/conv1d_3/ReluRelu$sequential/conv1d_3/BiasAdd:output:0*
T0*$
_output_shapes
:?{?2
sequential/conv1d_3/Relu?
)sequential/max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential/max_pooling1d_1/ExpandDims/dim?
%sequential/max_pooling1d_1/ExpandDims
ExpandDims&sequential/conv1d_3/Relu:activations:02sequential/max_pooling1d_1/ExpandDims/dim:output:0*
T0*(
_output_shapes
:?{?2'
%sequential/max_pooling1d_1/ExpandDims?
"sequential/max_pooling1d_1/MaxPoolMaxPool.sequential/max_pooling1d_1/ExpandDims:output:0*(
_output_shapes
:?=?*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling1d_1/MaxPool?
"sequential/max_pooling1d_1/SqueezeSqueeze+sequential/max_pooling1d_1/MaxPool:output:0*
T0*$
_output_shapes
:?=?*
squeeze_dims
2$
"sequential/max_pooling1d_1/Squeeze?
sequential/dropout_1/IdentityIdentity+sequential/max_pooling1d_1/Squeeze:output:0*
T0*$
_output_shapes
:?=?2
sequential/dropout_1/Identity?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape&sequential/dropout_1/Identity:output:0!sequential/flatten/Const:output:0*
T0* 
_output_shapes
:
??=2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp3sequential_dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
?=?*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0* 
_output_shapes
:
??2
sequential/dense/Relu?
sequential/dropout_2/IdentityIdentity#sequential/dense/Relu:activations:0*
T0* 
_output_shapes
:
??2
sequential/dropout_2/Identity?
'sequential/Output/MatMul/ReadVariableOpReadVariableOp5sequential_output_matmul_readvariableop_output_kernel*
_output_shapes
:	?*
dtype02)
'sequential/Output/MatMul/ReadVariableOp?
sequential/Output/MatMulMatMul&sequential/dropout_2/Identity:output:0/sequential/Output/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/Output/MatMul?
(sequential/Output/BiasAdd/ReadVariableOpReadVariableOp4sequential_output_biasadd_readvariableop_output_bias*
_output_shapes
:*
dtype02*
(sequential/Output/BiasAdd/ReadVariableOp?
sequential/Output/BiasAddBiasAdd"sequential/Output/MatMul:product:00sequential/Output/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
sequential/Output/BiasAdd?
sequential/Output/SoftmaxSoftmax"sequential/Output/BiasAdd:output:0*
T0*
_output_shapes
:	?2
sequential/Output/Softmax?
IdentityIdentity#sequential/Output/Softmax:softmax:0)^sequential/Output/BiasAdd/ReadVariableOp(^sequential/Output/MatMul/ReadVariableOp)^sequential/conv1d/BiasAdd/ReadVariableOp5^sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp+^sequential/conv1d_1/BiasAdd/ReadVariableOp7^sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+^sequential/conv1d_2/BiasAdd/ReadVariableOp7^sequential/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+^sequential/conv1d_3/BiasAdd/ReadVariableOp7^sequential/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:???::::::::::::2T
(sequential/Output/BiasAdd/ReadVariableOp(sequential/Output/BiasAdd/ReadVariableOp2R
'sequential/Output/MatMul/ReadVariableOp'sequential/Output/MatMul/ReadVariableOp2T
(sequential/conv1d/BiasAdd/ReadVariableOp(sequential/conv1d/BiasAdd/ReadVariableOp2l
4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp2X
*sequential/conv1d_1/BiasAdd/ReadVariableOp*sequential/conv1d_1/BiasAdd/ReadVariableOp2p
6sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp6sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2X
*sequential/conv1d_2/BiasAdd/ReadVariableOp*sequential/conv1d_2/BiasAdd/ReadVariableOp2p
6sequential/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp6sequential/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2X
*sequential/conv1d_3/BiasAdd/ReadVariableOp*sequential/conv1d_3/BiasAdd/ReadVariableOp2p
6sequential/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp6sequential/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:N J
%
_output_shapes
:???
!
_user_specified_name	input_1
?

b
C__inference_dropout_2_layer_call_and_return_conditional_losses_4393

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constl
dropout/MulMulinputsdropout/Const:output:0*
T0* 
_output_shapes
:
??2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?  ?   2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0* 
_output_shapes
:
??*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0* 
_output_shapes
:
??2
dropout/GreaterEqualx
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
* 
_output_shapes
:
??2
dropout/Casts
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0* 
_output_shapes
:
??2
dropout/Mul_1^
IdentityIdentitydropout/Mul_1:z:0*
T0* 
_output_shapes
:
??2

Identity"
identityIdentity:output:0*
_input_shapes
:
??:H D
 
_output_shapes
:
??
 
_user_specified_nameinputs
?
?
B__inference_conv1d_3_layer_call_and_return_conditional_losses_4318

inputs6
2conv1d_expanddims_1_readvariableop_conv1d_3_kernel(
$biasadd_readvariableop_conv1d_3_bias
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*(
_output_shapes
:?}?2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_expanddims_1_readvariableop_conv1d_3_kernel*$
_output_shapes
:??*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:?{?*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*$
_output_shapes
:?{?*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv1d_3_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:?{?2	
BiasAddU
ReluReluBiasAdd:output:0*
T0*$
_output_shapes
:?{?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*$
_output_shapes
:?{?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?}?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:L H
$
_output_shapes
:?}?
 
_user_specified_nameinputs
?

`
A__inference_dropout_layer_call_and_return_conditional_losses_4264

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consto
dropout/MulMulinputsdropout/Const:output:0*
T0*#
_output_shapes
:?@2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"?     @   2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*#
_output_shapes
:?@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?@2
dropout/GreaterEqual{
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?@2
dropout/Castv
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*#
_output_shapes
:?@2
dropout/Mul_1a
IdentityIdentitydropout/Mul_1:z:0*
T0*#
_output_shapes
:?@2

Identity"
identityIdentity:output:0*"
_input_shapes
:?@:K G
#
_output_shapes
:?@
 
_user_specified_nameinputs
?	
?
@__inference_Output_layer_call_and_return_conditional_losses_4419

inputs'
#matmul_readvariableop_output_kernel&
"biasadd_readvariableop_output_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp#matmul_readvariableop_output_kernel*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_output_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2	
BiasAddY
SoftmaxSoftmaxBiasAdd:output:0*
T0*
_output_shapes
:	?2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*'
_input_shapes
:
??::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:H D
 
_output_shapes
:
??
 
_user_specified_nameinputs
?
e
I__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_3572

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?
 __inference__traced_restore_4717
file_prefix"
assignvariableop_conv1d_kernel"
assignvariableop_1_conv1d_bias&
"assignvariableop_2_conv1d_1_kernel$
 assignvariableop_3_conv1d_1_bias&
"assignvariableop_4_conv1d_2_kernel$
 assignvariableop_5_conv1d_2_bias&
"assignvariableop_6_conv1d_3_kernel$
 assignvariableop_7_conv1d_3_bias#
assignvariableop_8_dense_kernel!
assignvariableop_9_dense_bias%
!assignvariableop_10_output_kernel#
assignvariableop_11_output_bias*
&assignvariableop_12_training_adam_iter,
(assignvariableop_13_training_adam_beta_1,
(assignvariableop_14_training_adam_beta_2+
'assignvariableop_15_training_adam_decay3
/assignvariableop_16_training_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count5
1assignvariableop_19_training_adam_conv1d_kernel_m3
/assignvariableop_20_training_adam_conv1d_bias_m7
3assignvariableop_21_training_adam_conv1d_1_kernel_m5
1assignvariableop_22_training_adam_conv1d_1_bias_m7
3assignvariableop_23_training_adam_conv1d_2_kernel_m5
1assignvariableop_24_training_adam_conv1d_2_bias_m7
3assignvariableop_25_training_adam_conv1d_3_kernel_m5
1assignvariableop_26_training_adam_conv1d_3_bias_m4
0assignvariableop_27_training_adam_dense_kernel_m2
.assignvariableop_28_training_adam_dense_bias_m5
1assignvariableop_29_training_adam_output_kernel_m3
/assignvariableop_30_training_adam_output_bias_m5
1assignvariableop_31_training_adam_conv1d_kernel_v3
/assignvariableop_32_training_adam_conv1d_bias_v7
3assignvariableop_33_training_adam_conv1d_1_kernel_v5
1assignvariableop_34_training_adam_conv1d_1_bias_v7
3assignvariableop_35_training_adam_conv1d_2_kernel_v5
1assignvariableop_36_training_adam_conv1d_2_bias_v7
3assignvariableop_37_training_adam_conv1d_3_kernel_v5
1assignvariableop_38_training_adam_conv1d_3_bias_v4
0assignvariableop_39_training_adam_dense_kernel_v2
.assignvariableop_40_training_adam_dense_bias_v5
1assignvariableop_41_training_adam_output_kernel_v3
/assignvariableop_42_training_adam_output_bias_v
identity_44??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv1d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv1d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv1d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv1d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv1d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_output_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_output_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_training_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp(assignvariableop_13_training_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp(assignvariableop_14_training_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp'assignvariableop_15_training_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp/assignvariableop_16_training_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp1assignvariableop_19_training_adam_conv1d_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp/assignvariableop_20_training_adam_conv1d_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp3assignvariableop_21_training_adam_conv1d_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp1assignvariableop_22_training_adam_conv1d_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp3assignvariableop_23_training_adam_conv1d_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp1assignvariableop_24_training_adam_conv1d_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp3assignvariableop_25_training_adam_conv1d_3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp1assignvariableop_26_training_adam_conv1d_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp0assignvariableop_27_training_adam_dense_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp.assignvariableop_28_training_adam_dense_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp1assignvariableop_29_training_adam_output_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp/assignvariableop_30_training_adam_output_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp1assignvariableop_31_training_adam_conv1d_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp/assignvariableop_32_training_adam_conv1d_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp3assignvariableop_33_training_adam_conv1d_1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp1assignvariableop_34_training_adam_conv1d_1_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp3assignvariableop_35_training_adam_conv1d_2_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp1assignvariableop_36_training_adam_conv1d_2_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp3assignvariableop_37_training_adam_conv1d_3_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp1assignvariableop_38_training_adam_conv1d_3_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp0assignvariableop_39_training_adam_dense_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp.assignvariableop_40_training_adam_dense_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp1assignvariableop_41_training_adam_output_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp/assignvariableop_42_training_adam_output_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_429
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_43?
Identity_44IdentityIdentity_43:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_44"#
identity_44Identity_44:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422(
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
?
e
I__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_3561

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
@__inference_conv1d_layer_call_and_return_conditional_losses_3595

inputs4
0conv1d_expanddims_1_readvariableop_conv1d_kernel&
"biasadd_readvariableop_conv1d_bias
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*)
_output_shapes
:???2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0conv1d_expanddims_1_readvariableop_conv1d_kernel*#
_output_shapes
:?@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:??@*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*$
_output_shapes
:??@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_conv1d_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:??@2	
BiasAddU
ReluReluBiasAdd:output:0*
T0*$
_output_shapes
:??@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*$
_output_shapes
:??@2

Identity"
identityIdentity:output:0*,
_input_shapes
:???::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:M I
%
_output_shapes
:???
 
_user_specified_nameinputs
?
?
B__inference_conv1d_1_layer_call_and_return_conditional_losses_3623

inputs6
2conv1d_expanddims_1_readvariableop_conv1d_1_kernel(
$biasadd_readvariableop_conv1d_1_bias
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*(
_output_shapes
:??@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_expanddims_1_readvariableop_conv1d_1_kernel*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:??@*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*$
_output_shapes
:??@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv1d_1_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:??@2	
BiasAddU
ReluReluBiasAdd:output:0*
T0*$
_output_shapes
:??@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*$
_output_shapes
:??@2

Identity"
identityIdentity:output:0*+
_input_shapes
:??@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:L H
$
_output_shapes
:??@
 
_user_specified_nameinputs
?^
?
__inference__traced_save_4578
file_prefix,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop,
(savev2_conv1d_3_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop1
-savev2_training_adam_iter_read_readvariableop	3
/savev2_training_adam_beta_1_read_readvariableop3
/savev2_training_adam_beta_2_read_readvariableop2
.savev2_training_adam_decay_read_readvariableop:
6savev2_training_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_training_adam_conv1d_kernel_m_read_readvariableop:
6savev2_training_adam_conv1d_bias_m_read_readvariableop>
:savev2_training_adam_conv1d_1_kernel_m_read_readvariableop<
8savev2_training_adam_conv1d_1_bias_m_read_readvariableop>
:savev2_training_adam_conv1d_2_kernel_m_read_readvariableop<
8savev2_training_adam_conv1d_2_bias_m_read_readvariableop>
:savev2_training_adam_conv1d_3_kernel_m_read_readvariableop<
8savev2_training_adam_conv1d_3_bias_m_read_readvariableop;
7savev2_training_adam_dense_kernel_m_read_readvariableop9
5savev2_training_adam_dense_bias_m_read_readvariableop<
8savev2_training_adam_output_kernel_m_read_readvariableop:
6savev2_training_adam_output_bias_m_read_readvariableop<
8savev2_training_adam_conv1d_kernel_v_read_readvariableop:
6savev2_training_adam_conv1d_bias_v_read_readvariableop>
:savev2_training_adam_conv1d_1_kernel_v_read_readvariableop<
8savev2_training_adam_conv1d_1_bias_v_read_readvariableop>
:savev2_training_adam_conv1d_2_kernel_v_read_readvariableop<
8savev2_training_adam_conv1d_2_bias_v_read_readvariableop>
:savev2_training_adam_conv1d_3_kernel_v_read_readvariableop<
8savev2_training_adam_conv1d_3_bias_v_read_readvariableop;
7savev2_training_adam_dense_kernel_v_read_readvariableop9
5savev2_training_adam_dense_bias_v_read_readvariableop<
8savev2_training_adam_output_kernel_v_read_readvariableop:
6savev2_training_adam_output_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop-savev2_training_adam_iter_read_readvariableop/savev2_training_adam_beta_1_read_readvariableop/savev2_training_adam_beta_2_read_readvariableop.savev2_training_adam_decay_read_readvariableop6savev2_training_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_training_adam_conv1d_kernel_m_read_readvariableop6savev2_training_adam_conv1d_bias_m_read_readvariableop:savev2_training_adam_conv1d_1_kernel_m_read_readvariableop8savev2_training_adam_conv1d_1_bias_m_read_readvariableop:savev2_training_adam_conv1d_2_kernel_m_read_readvariableop8savev2_training_adam_conv1d_2_bias_m_read_readvariableop:savev2_training_adam_conv1d_3_kernel_m_read_readvariableop8savev2_training_adam_conv1d_3_bias_m_read_readvariableop7savev2_training_adam_dense_kernel_m_read_readvariableop5savev2_training_adam_dense_bias_m_read_readvariableop8savev2_training_adam_output_kernel_m_read_readvariableop6savev2_training_adam_output_bias_m_read_readvariableop8savev2_training_adam_conv1d_kernel_v_read_readvariableop6savev2_training_adam_conv1d_bias_v_read_readvariableop:savev2_training_adam_conv1d_1_kernel_v_read_readvariableop8savev2_training_adam_conv1d_1_bias_v_read_readvariableop:savev2_training_adam_conv1d_2_kernel_v_read_readvariableop8savev2_training_adam_conv1d_2_bias_v_read_readvariableop:savev2_training_adam_conv1d_3_kernel_v_read_readvariableop8savev2_training_adam_conv1d_3_bias_v_read_readvariableop7savev2_training_adam_dense_kernel_v_read_readvariableop5savev2_training_adam_dense_bias_v_read_readvariableop8savev2_training_adam_output_kernel_v_read_readvariableop6savev2_training_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :?@:@:@@:@:@?:?:??:?:
?=?:?:	?:: : : : : : : :?@:@:@@:@:@?:?:??:?:
?=?:?:	?::?@:@:@@:@:@?:?:??:?:
?=?:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_output_shapes
:?@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:)%
#
_output_shapes
:@?:!

_output_shapes	
:?:*&
$
_output_shapes
:??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
?=?:!


_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::
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
: :

_output_shapes
: :

_output_shapes
: :)%
#
_output_shapes
:?@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:)%
#
_output_shapes
:@?:!

_output_shapes	
:?:*&
$
_output_shapes
:??:!

_output_shapes	
:?:&"
 
_output_shapes
:
?=?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::) %
#
_output_shapes
:?@: !

_output_shapes
:@:("$
"
_output_shapes
:@@: #

_output_shapes
:@:)$%
#
_output_shapes
:@?:!%

_output_shapes	
:?:*&&
$
_output_shapes
:??:!'

_output_shapes	
:?:&("
 
_output_shapes
:
?=?:!)

_output_shapes	
:?:%*!

_output_shapes
:	?: +

_output_shapes
::,

_output_shapes
: 
?	
?
)__inference_sequential_layer_call_fn_3974
input_1
conv1d_kernel
conv1d_bias
conv1d_1_kernel
conv1d_1_bias
conv1d_2_kernel
conv1d_2_bias
conv1d_3_kernel
conv1d_3_bias
dense_kernel

dense_bias
output_kernel
output_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_kernelconv1d_biasconv1d_1_kernelconv1d_1_biasconv1d_2_kernelconv1d_2_biasconv1d_3_kernelconv1d_3_biasdense_kernel
dense_biasoutput_kerneloutput_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_39592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:???::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
%
_output_shapes
:???
!
_user_specified_name	input_1
??
?	
D__inference_sequential_layer_call_and_return_conditional_losses_4093

inputs;
7conv1d_conv1d_expanddims_1_readvariableop_conv1d_kernel-
)conv1d_biasadd_readvariableop_conv1d_bias?
;conv1d_1_conv1d_expanddims_1_readvariableop_conv1d_1_kernel1
-conv1d_1_biasadd_readvariableop_conv1d_1_bias?
;conv1d_2_conv1d_expanddims_1_readvariableop_conv1d_2_kernel1
-conv1d_2_biasadd_readvariableop_conv1d_2_bias?
;conv1d_3_conv1d_expanddims_1_readvariableop_conv1d_3_kernel1
-conv1d_3_biasadd_readvariableop_conv1d_3_bias,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias.
*output_matmul_readvariableop_output_kernel-
)output_biasadd_readvariableop_output_bias
identity??Output/BiasAdd/ReadVariableOp?Output/MatMul/ReadVariableOp?conv1d/BiasAdd/ReadVariableOp?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?conv1d_1/BiasAdd/ReadVariableOp?+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?conv1d_2/BiasAdd/ReadVariableOp?+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?conv1d_3/BiasAdd/ReadVariableOp?+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDimsinputs%conv1d/conv1d/ExpandDims/dim:output:0*
T0*)
_output_shapes
:???2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp7conv1d_conv1d_expanddims_1_readvariableop_conv1d_kernel*#
_output_shapes
:?@*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?@2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:??@*
paddingVALID*
strides
2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*$
_output_shapes
:??@*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp)conv1d_biasadd_readvariableop_conv1d_bias*
_output_shapes
:@*
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:??@2
conv1d/BiasAddj
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*$
_output_shapes
:??@2
conv1d/Relu?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*(
_output_shapes
:??@2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;conv1d_1_conv1d_expanddims_1_readvariableop_conv1d_1_kernel*"
_output_shapes
:@@*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:??@*
paddingVALID*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*$
_output_shapes
:??@*
squeeze_dims

?????????2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp-conv1d_1_biasadd_readvariableop_conv1d_1_bias*
_output_shapes
:@*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:??@2
conv1d_1/BiasAddp
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*$
_output_shapes
:??@2
conv1d_1/Relu~
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
max_pooling1d/ExpandDims/dim?
max_pooling1d/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0%max_pooling1d/ExpandDims/dim:output:0*
T0*(
_output_shapes
:??@2
max_pooling1d/ExpandDims?
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*'
_output_shapes
:?@*
ksize
*
paddingVALID*
strides
2
max_pooling1d/MaxPool?
max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*#
_output_shapes
:?@*
squeeze_dims
2
max_pooling1d/Squeezes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const?
dropout/dropout/MulMulmax_pooling1d/Squeeze:output:0dropout/dropout/Const:output:0*
T0*#
_output_shapes
:?@2
dropout/dropout/Mul?
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"?     @   2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*#
_output_shapes
:?@*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:?@2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
:?@2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*#
_output_shapes
:?@2
dropout/dropout/Mul_1?
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_2/conv1d/ExpandDims/dim?
conv1d_2/conv1d/ExpandDims
ExpandDimsdropout/dropout/Mul_1:z:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?@2
conv1d_2/conv1d/ExpandDims?
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;conv1d_2_conv1d_expanddims_1_readvariableop_conv1d_2_kernel*#
_output_shapes
:@?*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:?}?*
paddingVALID*
strides
2
conv1d_2/conv1d?
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*$
_output_shapes
:?}?*
squeeze_dims

?????????2
conv1d_2/conv1d/Squeeze?
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp-conv1d_2_biasadd_readvariableop_conv1d_2_bias*
_output_shapes	
:?*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp?
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:?}?2
conv1d_2/BiasAddp
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*$
_output_shapes
:?}?2
conv1d_2/Relu?
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_3/conv1d/ExpandDims/dim?
conv1d_3/conv1d/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*(
_output_shapes
:?}?2
conv1d_3/conv1d/ExpandDims?
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;conv1d_3_conv1d_expanddims_1_readvariableop_conv1d_3_kernel*$
_output_shapes
:??*
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dim?
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_3/conv1d/ExpandDims_1?
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*(
_output_shapes
:?{?*
paddingVALID*
strides
2
conv1d_3/conv1d?
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*$
_output_shapes
:?{?*
squeeze_dims

?????????2
conv1d_3/conv1d/Squeeze?
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp-conv1d_3_biasadd_readvariableop_conv1d_3_bias*
_output_shapes	
:?*
dtype02!
conv1d_3/BiasAdd/ReadVariableOp?
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:?{?2
conv1d_3/BiasAddp
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*$
_output_shapes
:?{?2
conv1d_3/Relu?
max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_1/ExpandDims/dim?
max_pooling1d_1/ExpandDims
ExpandDimsconv1d_3/Relu:activations:0'max_pooling1d_1/ExpandDims/dim:output:0*
T0*(
_output_shapes
:?{?2
max_pooling1d_1/ExpandDims?
max_pooling1d_1/MaxPoolMaxPool#max_pooling1d_1/ExpandDims:output:0*(
_output_shapes
:?=?*
ksize
*
paddingVALID*
strides
2
max_pooling1d_1/MaxPool?
max_pooling1d_1/SqueezeSqueeze max_pooling1d_1/MaxPool:output:0*
T0*$
_output_shapes
:?=?*
squeeze_dims
2
max_pooling1d_1/Squeezew
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Const?
dropout_1/dropout/MulMul max_pooling1d_1/Squeeze:output:0 dropout_1/dropout/Const:output:0*
T0*$
_output_shapes
:?=?2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"?  =   ?   2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*$
_output_shapes
:?=?*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*$
_output_shapes
:?=?2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*$
_output_shapes
:?=?2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*$
_output_shapes
:?=?2
dropout_1/dropout/Mul_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapedropout_1/dropout/Mul_1:z:0flatten/Const:output:0*
T0* 
_output_shapes
:
??=2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
?=?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
dense/BiasAddc

dense/ReluReludense/BiasAdd:output:0*
T0* 
_output_shapes
:
??2

dense/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/dropout/Const?
dropout_2/dropout/MulMuldense/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0* 
_output_shapes
:
??2
dropout_2/dropout/Mul?
dropout_2/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"?  ?   2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0* 
_output_shapes
:
??*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0* 
_output_shapes
:
??2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
* 
_output_shapes
:
??2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0* 
_output_shapes
:
??2
dropout_2/dropout/Mul_1?
Output/MatMul/ReadVariableOpReadVariableOp*output_matmul_readvariableop_output_kernel*
_output_shapes
:	?*
dtype02
Output/MatMul/ReadVariableOp?
Output/MatMulMatMuldropout_2/dropout/Mul_1:z:0$Output/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
Output/MatMul?
Output/BiasAdd/ReadVariableOpReadVariableOp)output_biasadd_readvariableop_output_bias*
_output_shapes
:*
dtype02
Output/BiasAdd/ReadVariableOp?
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
Output/BiasAddn
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*
_output_shapes
:	?2
Output/Softmax?
IdentityIdentityOutput/Softmax:softmax:0^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/conv1d/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:???::::::::::::2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:M I
%
_output_shapes
:???
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
9
input_1.
serving_default_input_1:0???2
Output(
StatefulPartitionedCall:0	?tensorflow/serving/predict:??
?W
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?S
_tf_keras_sequential?S{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [500, 258, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "Output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [500, 258, 128]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [500, 258, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "Output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "loss_weights": null, "sample_weight_mode": null, "weighted_metrics": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [500, 258, 128]}}
?	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [500, 256, 64]}}
?
regularization_losses
 trainable_variables
!	variables
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling1D", "name": "max_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
#regularization_losses
$trainable_variables
%	variables
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?	

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [500, 127, 64]}}
?	

-kernel
.bias
/regularization_losses
0trainable_variables
1	variables
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [500, 125, 128]}}
?
3regularization_losses
4trainable_variables
5	variables
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling1D", "name": "max_pooling1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
7regularization_losses
8trainable_variables
9	variables
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?
;regularization_losses
<trainable_variables
=	variables
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

?kernel
@bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7808}}}, "build_input_shape": {"class_name": "TensorShape", "items": [500, 7808]}}
?
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?

Ikernel
Jbias
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "Output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [500, 128]}}
?
Oiter

Pbeta_1

Qbeta_2
	Rdecay
Slearning_ratem?m?m?m?'m?(m?-m?.m??m?@m?Im?Jm?v?v?v?v?'v?(v?-v?.v??v?@v?Iv?Jv?"
	optimizer
 "
trackable_list_wrapper
v
0
1
2
3
'4
(5
-6
.7
?8
@9
I10
J11"
trackable_list_wrapper
v
0
1
2
3
'4
(5
-6
.7
?8
@9
I10
J11"
trackable_list_wrapper
?
Tlayer_regularization_losses
Unon_trainable_variables

Vlayers
Wlayer_metrics
regularization_losses
trainable_variables
	variables
Xmetrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
$:"?@2conv1d/kernel
:@2conv1d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Ylayer_regularization_losses

Zlayers
[non_trainable_variables
\layer_metrics
regularization_losses
trainable_variables
	variables
]metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#@@2conv1d_1/kernel
:@2conv1d_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
^layer_regularization_losses

_layers
`non_trainable_variables
alayer_metrics
regularization_losses
trainable_variables
	variables
bmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
clayer_regularization_losses

dlayers
enon_trainable_variables
flayer_metrics
regularization_losses
 trainable_variables
!	variables
gmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
hlayer_regularization_losses

ilayers
jnon_trainable_variables
klayer_metrics
#regularization_losses
$trainable_variables
%	variables
lmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@?2conv1d_2/kernel
:?2conv1d_2/bias
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
mlayer_regularization_losses

nlayers
onon_trainable_variables
player_metrics
)regularization_losses
*trainable_variables
+	variables
qmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%??2conv1d_3/kernel
:?2conv1d_3/bias
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
?
rlayer_regularization_losses

slayers
tnon_trainable_variables
ulayer_metrics
/regularization_losses
0trainable_variables
1	variables
vmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
wlayer_regularization_losses

xlayers
ynon_trainable_variables
zlayer_metrics
3regularization_losses
4trainable_variables
5	variables
{metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
|layer_regularization_losses

}layers
~non_trainable_variables
layer_metrics
7regularization_losses
8trainable_variables
9	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?non_trainable_variables
?layer_metrics
;regularization_losses
<trainable_variables
=	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :
?=?2dense/kernel
:?2
dense/bias
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?non_trainable_variables
?layer_metrics
Aregularization_losses
Btrainable_variables
C	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?non_trainable_variables
?layer_metrics
Eregularization_losses
Ftrainable_variables
G	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	?2Output/kernel
:2Output/bias
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?non_trainable_variables
?layer_metrics
Kregularization_losses
Ltrainable_variables
M	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2training/Adam/iter
: (2training/Adam/beta_1
: (2training/Adam/beta_2
: (2training/Adam/decay
%:# (2training/Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
2:0?@2training/Adam/conv1d/kernel/m
':%@2training/Adam/conv1d/bias/m
3:1@@2training/Adam/conv1d_1/kernel/m
):'@2training/Adam/conv1d_1/bias/m
4:2@?2training/Adam/conv1d_2/kernel/m
*:(?2training/Adam/conv1d_2/bias/m
5:3??2training/Adam/conv1d_3/kernel/m
*:(?2training/Adam/conv1d_3/bias/m
.:,
?=?2training/Adam/dense/kernel/m
':%?2training/Adam/dense/bias/m
.:,	?2training/Adam/Output/kernel/m
':%2training/Adam/Output/bias/m
2:0?@2training/Adam/conv1d/kernel/v
':%@2training/Adam/conv1d/bias/v
3:1@@2training/Adam/conv1d_1/kernel/v
):'@2training/Adam/conv1d_1/bias/v
4:2@?2training/Adam/conv1d_2/kernel/v
*:(?2training/Adam/conv1d_2/bias/v
5:3??2training/Adam/conv1d_3/kernel/v
*:(?2training/Adam/conv1d_3/bias/v
.:,
?=?2training/Adam/dense/kernel/v
':%?2training/Adam/dense/bias/v
.:,	?2training/Adam/Output/kernel/v
':%2training/Adam/Output/bias/v
?2?
)__inference_sequential_layer_call_fn_4189
)__inference_sequential_layer_call_fn_3929
)__inference_sequential_layer_call_fn_4206
)__inference_sequential_layer_call_fn_3974?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_3529?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *$?!
?
input_1???
?2?
D__inference_sequential_layer_call_and_return_conditional_losses_3883
D__inference_sequential_layer_call_and_return_conditional_losses_3855
D__inference_sequential_layer_call_and_return_conditional_losses_4093
D__inference_sequential_layer_call_and_return_conditional_losses_4172?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference_conv1d_layer_call_fn_4229?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_conv1d_layer_call_and_return_conditional_losses_4222?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv1d_1_layer_call_fn_4252?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv1d_1_layer_call_and_return_conditional_losses_4245?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_max_pooling1d_layer_call_fn_3552?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_3538?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
&__inference_dropout_layer_call_fn_4274
&__inference_dropout_layer_call_fn_4279?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_dropout_layer_call_and_return_conditional_losses_4264
A__inference_dropout_layer_call_and_return_conditional_losses_4269?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_conv1d_2_layer_call_fn_4302?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv1d_2_layer_call_and_return_conditional_losses_4295?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv1d_3_layer_call_fn_4325?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv1d_3_layer_call_and_return_conditional_losses_4318?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_max_pooling1d_1_layer_call_fn_3575?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
I__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_3561?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
(__inference_dropout_1_layer_call_fn_4347
(__inference_dropout_1_layer_call_fn_4352?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dropout_1_layer_call_and_return_conditional_losses_4342
C__inference_dropout_1_layer_call_and_return_conditional_losses_4337?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_flatten_layer_call_fn_4363?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_flatten_layer_call_and_return_conditional_losses_4358?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_dense_layer_call_fn_4381?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_dense_layer_call_and_return_conditional_losses_4374?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dropout_2_layer_call_fn_4403
(__inference_dropout_2_layer_call_fn_4408?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dropout_2_layer_call_and_return_conditional_losses_4398
C__inference_dropout_2_layer_call_and_return_conditional_losses_4393?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference_Output_layer_call_fn_4426?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_Output_layer_call_and_return_conditional_losses_4419?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_3993input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
@__inference_Output_layer_call_and_return_conditional_losses_4419MIJ(?%
?
?
inputs
??
? "?
?
0	?
? i
%__inference_Output_layer_call_fn_4426@IJ(?%
?
?
inputs
??
? "?	??
__inference__wrapped_model_3529g'(-.?@IJ.?+
$?!
?
input_1???
? "'?$
"
Output?
Output	??
B__inference_conv1d_1_layer_call_and_return_conditional_losses_4245V,?)
"?
?
inputs??@
? ""?
?
0??@
? t
'__inference_conv1d_1_layer_call_fn_4252I,?)
"?
?
inputs??@
? "???@?
B__inference_conv1d_2_layer_call_and_return_conditional_losses_4295U'(+?(
!?
?
inputs?@
? ""?
?
0?}?
? s
'__inference_conv1d_2_layer_call_fn_4302H'(+?(
!?
?
inputs?@
? "??}??
B__inference_conv1d_3_layer_call_and_return_conditional_losses_4318V-.,?)
"?
?
inputs?}?
? ""?
?
0?{?
? t
'__inference_conv1d_3_layer_call_fn_4325I-.,?)
"?
?
inputs?}?
? "??{??
@__inference_conv1d_layer_call_and_return_conditional_losses_4222W-?*
#? 
?
inputs???
? ""?
?
0??@
? s
%__inference_conv1d_layer_call_fn_4229J-?*
#? 
?
inputs???
? "???@?
?__inference_dense_layer_call_and_return_conditional_losses_4374N?@(?%
?
?
inputs
??=
? "?
?
0
??
? i
$__inference_dense_layer_call_fn_4381A?@(?%
?
?
inputs
??=
? "?
???
C__inference_dropout_1_layer_call_and_return_conditional_losses_4337V0?-
&?#
?
inputs?=?
p
? ""?
?
0?=?
? ?
C__inference_dropout_1_layer_call_and_return_conditional_losses_4342V0?-
&?#
?
inputs?=?
p 
? ""?
?
0?=?
? u
(__inference_dropout_1_layer_call_fn_4347I0?-
&?#
?
inputs?=?
p
? "??=?u
(__inference_dropout_1_layer_call_fn_4352I0?-
&?#
?
inputs?=?
p 
? "??=??
C__inference_dropout_2_layer_call_and_return_conditional_losses_4393N,?)
"?
?
inputs
??
p
? "?
?
0
??
? ?
C__inference_dropout_2_layer_call_and_return_conditional_losses_4398N,?)
"?
?
inputs
??
p 
? "?
?
0
??
? m
(__inference_dropout_2_layer_call_fn_4403A,?)
"?
?
inputs
??
p
? "?
??m
(__inference_dropout_2_layer_call_fn_4408A,?)
"?
?
inputs
??
p 
? "?
???
A__inference_dropout_layer_call_and_return_conditional_losses_4264T/?,
%?"
?
inputs?@
p
? "!?
?
0?@
? ?
A__inference_dropout_layer_call_and_return_conditional_losses_4269T/?,
%?"
?
inputs?@
p 
? "!?
?
0?@
? q
&__inference_dropout_layer_call_fn_4274G/?,
%?"
?
inputs?@
p
? "??@q
&__inference_dropout_layer_call_fn_4279G/?,
%?"
?
inputs?@
p 
? "??@?
A__inference_flatten_layer_call_and_return_conditional_losses_4358N,?)
"?
?
inputs?=?
? "?
?
0
??=
? k
&__inference_flatten_layer_call_fn_4363A,?)
"?
?
inputs?=?
? "?
??=?
I__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_3561?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
.__inference_max_pooling1d_1_layer_call_fn_3575wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_3538?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
,__inference_max_pooling1d_layer_call_fn_3552wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
D__inference_sequential_layer_call_and_return_conditional_losses_3855e'(-.?@IJ6?3
,?)
?
input_1???
p

 
? "?
?
0	?
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_3883e'(-.?@IJ6?3
,?)
?
input_1???
p 

 
? "?
?
0	?
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_4093d'(-.?@IJ5?2
+?(
?
inputs???
p

 
? "?
?
0	?
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_4172d'(-.?@IJ5?2
+?(
?
inputs???
p 

 
? "?
?
0	?
? ?
)__inference_sequential_layer_call_fn_3929X'(-.?@IJ6?3
,?)
?
input_1???
p

 
? "?	??
)__inference_sequential_layer_call_fn_3974X'(-.?@IJ6?3
,?)
?
input_1???
p 

 
? "?	??
)__inference_sequential_layer_call_fn_4189W'(-.?@IJ5?2
+?(
?
inputs???
p

 
? "?	??
)__inference_sequential_layer_call_fn_4206W'(-.?@IJ5?2
+?(
?
inputs???
p 

 
? "?	??
"__inference_signature_wrapper_3993r'(-.?@IJ9?6
? 
/?,
*
input_1?
input_1???"'?$
"
Output?
Output	?