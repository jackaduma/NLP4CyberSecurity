¼Ø
Æ
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
¾
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
executor_typestring 
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
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8³
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	d*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:d*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:d*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
w
Output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¨(*
shared_nameOutput/kernel
p
!Output/kernel/Read/ReadVariableOpReadVariableOpOutput/kernel*
_output_shapes
:	¨(*
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

training/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nametraining/Adam/learning_rate

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

training/Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*-
shared_nametraining/Adam/dense/kernel/m

0training/Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/m*
_output_shapes
:	d*
dtype0

training/Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*+
shared_nametraining/Adam/dense/bias/m

.training/Adam/dense/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/m*
_output_shapes
:d*
dtype0

training/Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*/
shared_name training/Adam/dense_1/kernel/m

2training/Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/kernel/m*
_output_shapes

:d*
dtype0

training/Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nametraining/Adam/dense_1/bias/m

0training/Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/bias/m*
_output_shapes
:*
dtype0

training/Adam/Output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¨(*.
shared_nametraining/Adam/Output/kernel/m

1training/Adam/Output/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/Output/kernel/m*
_output_shapes
:	¨(*
dtype0

training/Adam/Output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nametraining/Adam/Output/bias/m

/training/Adam/Output/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/Output/bias/m*
_output_shapes
:*
dtype0

training/Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*-
shared_nametraining/Adam/dense/kernel/v

0training/Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/v*
_output_shapes
:	d*
dtype0

training/Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*+
shared_nametraining/Adam/dense/bias/v

.training/Adam/dense/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/v*
_output_shapes
:d*
dtype0

training/Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*/
shared_name training/Adam/dense_1/kernel/v

2training/Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/kernel/v*
_output_shapes

:d*
dtype0

training/Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nametraining/Adam/dense_1/bias/v

0training/Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/bias/v*
_output_shapes
:*
dtype0

training/Adam/Output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¨(*.
shared_nametraining/Adam/Output/kernel/v

1training/Adam/Output/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/Output/kernel/v*
_output_shapes
:	¨(*
dtype0

training/Adam/Output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nametraining/Adam/Output/bias/v

/training/Adam/Output/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/Output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ô*
valueÊ*BÇ* BÀ*

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
	optimizer
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
 	keras_api
R
!trainable_variables
"regularization_losses
#	variables
$	keras_api
h

%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
¬
+iter

,beta_1

-beta_2
	.decay
/learning_ratemYmZm[m\%m]&m^v_v`vavb%vc&vd
*
0
1
2
3
%4
&5
 
*
0
1
2
3
%4
&5
­

0layers
trainable_variables
1layer_regularization_losses
2non_trainable_variables
	regularization_losses

	variables
3layer_metrics
4metrics
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

5layers
trainable_variables
6layer_regularization_losses
7non_trainable_variables
regularization_losses
	variables
8layer_metrics
9metrics
 
 
 
­

:layers
trainable_variables
;layer_regularization_losses
<non_trainable_variables
regularization_losses
	variables
=layer_metrics
>metrics
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

?layers
trainable_variables
@layer_regularization_losses
Anon_trainable_variables
regularization_losses
	variables
Blayer_metrics
Cmetrics
 
 
 
­

Dlayers
trainable_variables
Elayer_regularization_losses
Fnon_trainable_variables
regularization_losses
	variables
Glayer_metrics
Hmetrics
 
 
 
­

Ilayers
!trainable_variables
Jlayer_regularization_losses
Knon_trainable_variables
"regularization_losses
#	variables
Llayer_metrics
Mmetrics
YW
VARIABLE_VALUEOutput/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEOutput/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1
 

%0
&1
­

Nlayers
'trainable_variables
Olayer_regularization_losses
Pnon_trainable_variables
(regularization_losses
)	variables
Qlayer_metrics
Rmetrics
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
*
0
1
2
3
4
5
 
 
 

S0
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
D
	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

T0
U1

W	variables

VARIABLE_VALUEtraining/Adam/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEtraining/Adam/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEtraining/Adam/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEtraining/Adam/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEtraining/Adam/Output/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEtraining/Adam/Output/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEtraining/Adam/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEtraining/Adam/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEtraining/Adam/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEtraining/Adam/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEtraining/Adam/Output/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEtraining/Adam/Output/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
v
serving_default_input_1Placeholder*%
_output_shapes
:ô*
dtype0*
shape:ô

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/biasOutput/kernelOutput/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ô*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_1689
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
â

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp!Output/kernel/Read/ReadVariableOpOutput/bias/Read/ReadVariableOp&training/Adam/iter/Read/ReadVariableOp(training/Adam/beta_1/Read/ReadVariableOp(training/Adam/beta_2/Read/ReadVariableOp'training/Adam/decay/Read/ReadVariableOp/training/Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp0training/Adam/dense/kernel/m/Read/ReadVariableOp.training/Adam/dense/bias/m/Read/ReadVariableOp2training/Adam/dense_1/kernel/m/Read/ReadVariableOp0training/Adam/dense_1/bias/m/Read/ReadVariableOp1training/Adam/Output/kernel/m/Read/ReadVariableOp/training/Adam/Output/bias/m/Read/ReadVariableOp0training/Adam/dense/kernel/v/Read/ReadVariableOp.training/Adam/dense/bias/v/Read/ReadVariableOp2training/Adam/dense_1/kernel/v/Read/ReadVariableOp0training/Adam/dense_1/bias/v/Read/ReadVariableOp1training/Adam/Output/kernel/v/Read/ReadVariableOp/training/Adam/Output/bias/v/Read/ReadVariableOpConst*&
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
GPU 2J 8 *&
f!R
__inference__traced_save_2024
é
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasOutput/kernelOutput/biastraining/Adam/itertraining/Adam/beta_1training/Adam/beta_2training/Adam/decaytraining/Adam/learning_ratetotalcounttraining/Adam/dense/kernel/mtraining/Adam/dense/bias/mtraining/Adam/dense_1/kernel/mtraining/Adam/dense_1/bias/mtraining/Adam/Output/kernel/mtraining/Adam/Output/bias/mtraining/Adam/dense/kernel/vtraining/Adam/dense/bias/vtraining/Adam/dense_1/kernel/vtraining/Adam/dense_1/bias/vtraining/Adam/Output/kernel/vtraining/Adam/Output/bias/v*%
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
GPU 2J 8 *)
f$R"
 __inference__traced_restore_2109

]
A__inference_flatten_layer_call_and_return_conditional_losses_1573

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(  2
Const`
ReshapeReshapeinputsConst:output:0*
T0* 
_output_shapes
:
ô¨(2	
Reshape]
IdentityIdentityReshape:output:0*
T0* 
_output_shapes
:
ô¨(2

Identity"
identityIdentity:output:0*#
_input_shapes
:ô:L H
$
_output_shapes
:ô
 
_user_specified_nameinputs
º
ä
?__inference_dense_layer_call_and_return_conditional_losses_1814

inputs)
%tensordot_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp%tensordot_readvariableop_dense_kernel*
_output_shapes
:	d*
dtype02
Tensordot/ReadVariableOp
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"è÷    2
Tensordot/Reshape/shape
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0*!
_output_shapes
:èï2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
èïd2
Tensordot/MatMulw
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ô    d   2
Tensordot/shape
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*$
_output_shapes
:ôd2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:ôd2	
BiasAddU
ReluReluBiasAdd:output:0*
T0*$
_output_shapes
:ôd2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*$
_output_shapes
:ôd2

Identity"
identityIdentity:output:0*,
_input_shapes
:ô::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:M I
%
_output_shapes
:ô
 
_user_specified_nameinputs

`
A__inference_dropout_layer_call_and_return_conditional_losses_1492

inputs
identityc
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
:ôd2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"ô    d   2
dropout/Shape±
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*$
_output_shapes
:ôd*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y»
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*$
_output_shapes
:ôd2
dropout/GreaterEqual|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*$
_output_shapes
:ôd2
dropout/Castw
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*$
_output_shapes
:ôd2
dropout/Mul_1b
IdentityIdentitydropout/Mul_1:z:0*
T0*$
_output_shapes
:ôd2

Identity"
identityIdentity:output:0*#
_input_shapes
:ôd:L H
$
_output_shapes
:ôd
 
_user_specified_nameinputs

B
&__inference_dropout_layer_call_fn_1848

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ôd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_14972
PartitionedCalli
IdentityIdentityPartitionedCall:output:0*
T0*$
_output_shapes
:ôd2

Identity"
identityIdentity:output:0*#
_input_shapes
:ôd:L H
$
_output_shapes
:ôd
 
_user_specified_nameinputs
Ê<
­
__inference__traced_save_2024
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop1
-savev2_training_adam_iter_read_readvariableop	3
/savev2_training_adam_beta_1_read_readvariableop3
/savev2_training_adam_beta_2_read_readvariableop2
.savev2_training_adam_decay_read_readvariableop:
6savev2_training_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop;
7savev2_training_adam_dense_kernel_m_read_readvariableop9
5savev2_training_adam_dense_bias_m_read_readvariableop=
9savev2_training_adam_dense_1_kernel_m_read_readvariableop;
7savev2_training_adam_dense_1_bias_m_read_readvariableop<
8savev2_training_adam_output_kernel_m_read_readvariableop:
6savev2_training_adam_output_bias_m_read_readvariableop;
7savev2_training_adam_dense_kernel_v_read_readvariableop9
5savev2_training_adam_dense_bias_v_read_readvariableop=
9savev2_training_adam_dense_1_kernel_v_read_readvariableop;
7savev2_training_adam_dense_1_bias_v_read_readvariableop<
8savev2_training_adam_output_kernel_v_read_readvariableop:
6savev2_training_adam_output_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*²
value¨B¥B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¼
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices²
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop-savev2_training_adam_iter_read_readvariableop/savev2_training_adam_beta_1_read_readvariableop/savev2_training_adam_beta_2_read_readvariableop.savev2_training_adam_decay_read_readvariableop6savev2_training_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop7savev2_training_adam_dense_kernel_m_read_readvariableop5savev2_training_adam_dense_bias_m_read_readvariableop9savev2_training_adam_dense_1_kernel_m_read_readvariableop7savev2_training_adam_dense_1_bias_m_read_readvariableop8savev2_training_adam_output_kernel_m_read_readvariableop6savev2_training_adam_output_bias_m_read_readvariableop7savev2_training_adam_dense_kernel_v_read_readvariableop5savev2_training_adam_dense_bias_v_read_readvariableop9savev2_training_adam_dense_1_kernel_v_read_readvariableop7savev2_training_adam_dense_1_bias_v_read_readvariableop8savev2_training_adam_output_kernel_v_read_readvariableop6savev2_training_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*½
_input_shapes«
¨: :	d:d:d::	¨(:: : : : : : : :	d:d:d::	¨(::	d:d:d::	¨(:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	¨(: 

_output_shapes
::
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
: :%!

_output_shapes
:	d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	¨(: 

_output_shapes
::%!

_output_shapes
:	d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	¨(: 

_output_shapes
::

_output_shapes
: 
ô
Ï
)__inference_sequential_layer_call_fn_1649
input_1
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
output_kernel
output_bias
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinput_1dense_kernel
dense_biasdense_1_kerneldense_1_biasoutput_kerneloutput_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ô*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_16402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	ô2

Identity"
identityIdentity:output:0*<
_input_shapes+
):ô::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
%
_output_shapes
:ô
!
_user_specified_name	input_1
ß

&__inference_dense_1_layer_call_fn_1870

inputs
dense_1_kernel
dense_1_bias
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_kerneldense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_15252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*$
_output_shapes
:ô2

Identity"
identityIdentity:output:0*+
_input_shapes
:ôd::22
StatefulPartitionedCallStatefulPartitionedCall:L H
$
_output_shapes
:ôd
 
_user_specified_nameinputs
+
É
D__inference_sequential_layer_call_and_return_conditional_losses_1777

inputs/
+dense_tensordot_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias3
/dense_1_tensordot_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias.
*output_matmul_readvariableop_output_kernel-
)output_biasadd_readvariableop_output_bias
identity¢Output/BiasAdd/ReadVariableOp¢Output/MatMul/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/Tensordot/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢ dense_1/Tensordot/ReadVariableOp­
dense/Tensordot/ReadVariableOpReadVariableOp+dense_tensordot_readvariableop_dense_kernel*
_output_shapes
:	d*
dtype02 
dense/Tensordot/ReadVariableOp
dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"è÷    2
dense/Tensordot/Reshape/shape
dense/Tensordot/ReshapeReshapeinputs&dense/Tensordot/Reshape/shape:output:0*
T0*!
_output_shapes
:èï2
dense/Tensordot/Reshape¯
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
èïd2
dense/Tensordot/MatMul
dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ô    d   2
dense/Tensordot/shape
dense/TensordotReshape dense/Tensordot/MatMul:product:0dense/Tensordot/shape:output:0*
T0*$
_output_shapes
:ôd2
dense/Tensordot 
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:d*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:ôd2
dense/BiasAddg

dense/ReluReludense/BiasAdd:output:0*
T0*$
_output_shapes
:ôd2

dense/Reluy
dropout/IdentityIdentitydense/Relu:activations:0*
T0*$
_output_shapes
:ôd2
dropout/Identity´
 dense_1/Tensordot/ReadVariableOpReadVariableOp/dense_1_tensordot_readvariableop_dense_1_kernel*
_output_shapes

:d*
dtype02"
 dense_1/Tensordot/ReadVariableOp
dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"è÷ d   2!
dense_1/Tensordot/Reshape/shape±
dense_1/Tensordot/ReshapeReshapedropout/Identity:output:0(dense_1/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
èïd2
dense_1/Tensordot/Reshape·
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
èï2
dense_1/Tensordot/MatMul
dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ô       2
dense_1/Tensordot/shape¦
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0 dense_1/Tensordot/shape:output:0*
T0*$
_output_shapes
:ô2
dense_1/Tensordot¨
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp 
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:ô2
dense_1/BiasAddm
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*$
_output_shapes
:ô2
dense_1/Relu
dropout_1/IdentityIdentitydense_1/Relu:activations:0*
T0*$
_output_shapes
:ô2
dropout_1/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(  2
flatten/Const
flatten/ReshapeReshapedropout_1/Identity:output:0flatten/Const:output:0*
T0* 
_output_shapes
:
ô¨(2
flatten/Reshape¨
Output/MatMul/ReadVariableOpReadVariableOp*output_matmul_readvariableop_output_kernel*
_output_shapes
:	¨(*
dtype02
Output/MatMul/ReadVariableOp
Output/MatMulMatMulflatten/Reshape:output:0$Output/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ô2
Output/MatMul¤
Output/BiasAdd/ReadVariableOpReadVariableOp)output_biasadd_readvariableop_output_bias*
_output_shapes
:*
dtype02
Output/BiasAdd/ReadVariableOp
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ô2
Output/BiasAddn
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*
_output_shapes
:	ô2
Output/Softmax§
IdentityIdentityOutput/Softmax:softmax:0^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp*
T0*
_output_shapes
:	ô2

Identity"
identityIdentity:output:0*<
_input_shapes+
):ô::::::2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp:M I
%
_output_shapes
:ô
 
_user_specified_nameinputs
¸
_
A__inference_dropout_layer_call_and_return_conditional_losses_1497

inputs

identity_1W
IdentityIdentityinputs*
T0*$
_output_shapes
:ôd2

Identityf

Identity_1IdentityIdentity:output:0*
T0*$
_output_shapes
:ôd2

Identity_1"!

identity_1Identity_1:output:0*#
_input_shapes
:ôd:L H
$
_output_shapes
:ôd
 
_user_specified_nameinputs
Ç=
É
D__inference_sequential_layer_call_and_return_conditional_losses_1740

inputs/
+dense_tensordot_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias3
/dense_1_tensordot_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias.
*output_matmul_readvariableop_output_kernel-
)output_biasadd_readvariableop_output_bias
identity¢Output/BiasAdd/ReadVariableOp¢Output/MatMul/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/Tensordot/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢ dense_1/Tensordot/ReadVariableOp­
dense/Tensordot/ReadVariableOpReadVariableOp+dense_tensordot_readvariableop_dense_kernel*
_output_shapes
:	d*
dtype02 
dense/Tensordot/ReadVariableOp
dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"è÷    2
dense/Tensordot/Reshape/shape
dense/Tensordot/ReshapeReshapeinputs&dense/Tensordot/Reshape/shape:output:0*
T0*!
_output_shapes
:èï2
dense/Tensordot/Reshape¯
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
èïd2
dense/Tensordot/MatMul
dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ô    d   2
dense/Tensordot/shape
dense/TensordotReshape dense/Tensordot/MatMul:product:0dense/Tensordot/shape:output:0*
T0*$
_output_shapes
:ôd2
dense/Tensordot 
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:d*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:ôd2
dense/BiasAddg

dense/ReluReludense/BiasAdd:output:0*
T0*$
_output_shapes
:ôd2

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*$
_output_shapes
:ôd2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"ô    d   2
dropout/dropout/ShapeÉ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*$
_output_shapes
:ôd*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yÛ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*$
_output_shapes
:ôd2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*$
_output_shapes
:ôd2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*$
_output_shapes
:ôd2
dropout/dropout/Mul_1´
 dense_1/Tensordot/ReadVariableOpReadVariableOp/dense_1_tensordot_readvariableop_dense_1_kernel*
_output_shapes

:d*
dtype02"
 dense_1/Tensordot/ReadVariableOp
dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"è÷ d   2!
dense_1/Tensordot/Reshape/shape±
dense_1/Tensordot/ReshapeReshapedropout/dropout/Mul_1:z:0(dense_1/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
èïd2
dense_1/Tensordot/Reshape·
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
èï2
dense_1/Tensordot/MatMul
dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ô       2
dense_1/Tensordot/shape¦
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0 dense_1/Tensordot/shape:output:0*
T0*$
_output_shapes
:ô2
dense_1/Tensordot¨
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp 
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:ô2
dense_1/BiasAddm
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*$
_output_shapes
:ô2
dense_1/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Const¢
dropout_1/dropout/MulMuldense_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*$
_output_shapes
:ô2
dropout_1/dropout/Mul
dropout_1/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"ô       2
dropout_1/dropout/ShapeÏ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*$
_output_shapes
:ô*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/yã
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*$
_output_shapes
:ô2 
dropout_1/dropout/GreaterEqual
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*$
_output_shapes
:ô2
dropout_1/dropout/Cast
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*$
_output_shapes
:ô2
dropout_1/dropout/Mul_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(  2
flatten/Const
flatten/ReshapeReshapedropout_1/dropout/Mul_1:z:0flatten/Const:output:0*
T0* 
_output_shapes
:
ô¨(2
flatten/Reshape¨
Output/MatMul/ReadVariableOpReadVariableOp*output_matmul_readvariableop_output_kernel*
_output_shapes
:	¨(*
dtype02
Output/MatMul/ReadVariableOp
Output/MatMulMatMulflatten/Reshape:output:0$Output/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ô2
Output/MatMul¤
Output/BiasAdd/ReadVariableOpReadVariableOp)output_biasadd_readvariableop_output_bias*
_output_shapes
:*
dtype02
Output/BiasAdd/ReadVariableOp
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ô2
Output/BiasAddn
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*
_output_shapes
:	ô2
Output/Softmax§
IdentityIdentityOutput/Softmax:softmax:0^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp*
T0*
_output_shapes
:	ô2

Identity"
identityIdentity:output:0*<
_input_shapes+
):ô::::::2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp:M I
%
_output_shapes
:ô
 
_user_specified_nameinputs
ô
Ï
)__inference_sequential_layer_call_fn_1676
input_1
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
output_kernel
output_bias
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinput_1dense_kernel
dense_biasdense_1_kerneldense_1_biasoutput_kerneloutput_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ô*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_16672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	ô2

Identity"
identityIdentity:output:0*<
_input_shapes+
):ô::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
%
_output_shapes
:ô
!
_user_specified_name	input_1

_
&__inference_dropout_layer_call_fn_1843

inputs
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ôd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_14922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*$
_output_shapes
:ôd2

Identity"
identityIdentity:output:0*#
_input_shapes
:ôd22
StatefulPartitionedCallStatefulPartitionedCall:L H
$
_output_shapes
:ôd
 
_user_specified_nameinputs

`
A__inference_dropout_layer_call_and_return_conditional_losses_1833

inputs
identityc
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
:ôd2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"ô    d   2
dropout/Shape±
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*$
_output_shapes
:ôd*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y»
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*$
_output_shapes
:ôd2
dropout/GreaterEqual|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*$
_output_shapes
:ôd2
dropout/Castw
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*$
_output_shapes
:ôd2
dropout/Mul_1b
IdentityIdentitydropout/Mul_1:z:0*
T0*$
_output_shapes
:ôd2

Identity"
identityIdentity:output:0*#
_input_shapes
:ôd:L H
$
_output_shapes
:ôd
 
_user_specified_nameinputs

D
(__inference_dropout_1_layer_call_fn_1897

inputs
identity¾
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_15542
PartitionedCalli
IdentityIdentityPartitionedCall:output:0*
T0*$
_output_shapes
:ô2

Identity"
identityIdentity:output:0*#
_input_shapes
:ô:L H
$
_output_shapes
:ô
 
_user_specified_nameinputs
â
Ü
D__inference_sequential_layer_call_and_return_conditional_losses_1667

inputs
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
output_output_kernel
output_output_bias
identity¢Output/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ôd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_14672
dense/StatefulPartitionedCallì
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ôd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_14972
dropout/PartitionedCall²
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_15252!
dense_1/StatefulPartitionedCallô
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_15542
dropout_1/PartitionedCallä
flatten/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
ô¨(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_15732
flatten/PartitionedCall¦
Output/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0output_output_kerneloutput_output_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_Output_layer_call_and_return_conditional_losses_15922 
Output/StatefulPartitionedCallÖ
IdentityIdentity'Output/StatefulPartitionedCall:output:0^Output/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*
_output_shapes
:	ô2

Identity"
identityIdentity:output:0*<
_input_shapes+
):ô::::::2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:M I
%
_output_shapes
:ô
 
_user_specified_nameinputs
º
ä
?__inference_dense_layer_call_and_return_conditional_losses_1467

inputs)
%tensordot_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp%tensordot_readvariableop_dense_kernel*
_output_shapes
:	d*
dtype02
Tensordot/ReadVariableOp
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"è÷    2
Tensordot/Reshape/shape
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0*!
_output_shapes
:èï2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
èïd2
Tensordot/MatMulw
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ô    d   2
Tensordot/shape
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*$
_output_shapes
:ôd2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:ôd2	
BiasAddU
ReluReluBiasAdd:output:0*
T0*$
_output_shapes
:ôd2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*$
_output_shapes
:ôd2

Identity"
identityIdentity:output:0*,
_input_shapes
:ô::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:M I
%
_output_shapes
:ô
 
_user_specified_nameinputs
È
È
"__inference_signature_wrapper_1689
input_1
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
output_kernel
output_bias
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1dense_kernel
dense_biasdense_1_kerneldense_1_biasoutput_kerneloutput_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ô*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_14482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	ô2

Identity"
identityIdentity:output:0*<
_input_shapes+
):ô::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
%
_output_shapes
:ô
!
_user_specified_name	input_1
À
ê
A__inference_dense_1_layer_call_and_return_conditional_losses_1525

inputs+
'tensordot_readvariableop_dense_1_kernel'
#biasadd_readvariableop_dense_1_bias
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_1_kernel*
_output_shapes

:d*
dtype02
Tensordot/ReadVariableOp
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"è÷ d   2
Tensordot/Reshape/shape
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
èïd2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
èï2
Tensordot/MatMulw
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ô       2
Tensordot/shape
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*$
_output_shapes
:ô2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:ô2	
BiasAddU
ReluReluBiasAdd:output:0*
T0*$
_output_shapes
:ô2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*$
_output_shapes
:ô2

Identity"
identityIdentity:output:0*+
_input_shapes
:ôd::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:L H
$
_output_shapes
:ôd
 
_user_specified_nameinputs

b
C__inference_dropout_1_layer_call_and_return_conditional_losses_1549

inputs
identityc
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
:ô2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"ô       2
dropout/Shape±
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*$
_output_shapes
:ô*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y»
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*$
_output_shapes
:ô2
dropout/GreaterEqual|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*$
_output_shapes
:ô2
dropout/Castw
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*$
_output_shapes
:ô2
dropout/Mul_1b
IdentityIdentitydropout/Mul_1:z:0*
T0*$
_output_shapes
:ô2

Identity"
identityIdentity:output:0*#
_input_shapes
:ô:L H
$
_output_shapes
:ô
 
_user_specified_nameinputs
Ô

$__inference_dense_layer_call_fn_1821

inputs
dense_kernel

dense_bias
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ôd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_14672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*$
_output_shapes
:ôd2

Identity"
identityIdentity:output:0*,
_input_shapes
:ô::22
StatefulPartitionedCallStatefulPartitionedCall:M I
%
_output_shapes
:ô
 
_user_specified_nameinputs
5
©
__inference__wrapped_model_1448
input_1:
6sequential_dense_tensordot_readvariableop_dense_kernel6
2sequential_dense_biasadd_readvariableop_dense_bias>
:sequential_dense_1_tensordot_readvariableop_dense_1_kernel:
6sequential_dense_1_biasadd_readvariableop_dense_1_bias9
5sequential_output_matmul_readvariableop_output_kernel8
4sequential_output_biasadd_readvariableop_output_bias
identity¢(sequential/Output/BiasAdd/ReadVariableOp¢'sequential/Output/MatMul/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢)sequential/dense/Tensordot/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢+sequential/dense_1/Tensordot/ReadVariableOpÎ
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp6sequential_dense_tensordot_readvariableop_dense_kernel*
_output_shapes
:	d*
dtype02+
)sequential/dense/Tensordot/ReadVariableOp¥
(sequential/dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"è÷    2*
(sequential/dense/Tensordot/Reshape/shape»
"sequential/dense/Tensordot/ReshapeReshapeinput_11sequential/dense/Tensordot/Reshape/shape:output:0*
T0*!
_output_shapes
:èï2$
"sequential/dense/Tensordot/ReshapeÛ
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
èïd2#
!sequential/dense/Tensordot/MatMul
 sequential/dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ô    d   2"
 sequential/dense/Tensordot/shapeÊ
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0)sequential/dense/Tensordot/shape:output:0*
T0*$
_output_shapes
:ôd2
sequential/dense/TensordotÁ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_biasadd_readvariableop_dense_bias*
_output_shapes
:d*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpÄ
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:ôd2
sequential/dense/BiasAdd
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*$
_output_shapes
:ôd2
sequential/dense/Relu
sequential/dropout/IdentityIdentity#sequential/dense/Relu:activations:0*
T0*$
_output_shapes
:ôd2
sequential/dropout/IdentityÕ
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp:sequential_dense_1_tensordot_readvariableop_dense_1_kernel*
_output_shapes

:d*
dtype02-
+sequential/dense_1/Tensordot/ReadVariableOp©
*sequential/dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"è÷ d   2,
*sequential/dense_1/Tensordot/Reshape/shapeÝ
$sequential/dense_1/Tensordot/ReshapeReshape$sequential/dropout/Identity:output:03sequential/dense_1/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
èïd2&
$sequential/dense_1/Tensordot/Reshapeã
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
èï2%
#sequential/dense_1/Tensordot/MatMul
"sequential/dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ô       2$
"sequential/dense_1/Tensordot/shapeÒ
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0+sequential/dense_1/Tensordot/shape:output:0*
T0*$
_output_shapes
:ô2
sequential/dense_1/TensordotÉ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp6sequential_dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpÌ
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:ô2
sequential/dense_1/BiasAdd
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*$
_output_shapes
:ô2
sequential/dense_1/Relu 
sequential/dropout_1/IdentityIdentity%sequential/dense_1/Relu:activations:0*
T0*$
_output_shapes
:ô2
sequential/dropout_1/Identity
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(  2
sequential/flatten/Const¹
sequential/flatten/ReshapeReshape&sequential/dropout_1/Identity:output:0!sequential/flatten/Const:output:0*
T0* 
_output_shapes
:
ô¨(2
sequential/flatten/ReshapeÉ
'sequential/Output/MatMul/ReadVariableOpReadVariableOp5sequential_output_matmul_readvariableop_output_kernel*
_output_shapes
:	¨(*
dtype02)
'sequential/Output/MatMul/ReadVariableOp¾
sequential/Output/MatMulMatMul#sequential/flatten/Reshape:output:0/sequential/Output/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ô2
sequential/Output/MatMulÅ
(sequential/Output/BiasAdd/ReadVariableOpReadVariableOp4sequential_output_biasadd_readvariableop_output_bias*
_output_shapes
:*
dtype02*
(sequential/Output/BiasAdd/ReadVariableOpÁ
sequential/Output/BiasAddBiasAdd"sequential/Output/MatMul:product:00sequential/Output/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ô2
sequential/Output/BiasAdd
sequential/Output/SoftmaxSoftmax"sequential/Output/BiasAdd:output:0*
T0*
_output_shapes
:	ô2
sequential/Output/Softmaxô
IdentityIdentity#sequential/Output/Softmax:softmax:0)^sequential/Output/BiasAdd/ReadVariableOp(^sequential/Output/MatMul/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*
T0*
_output_shapes
:	ô2

Identity"
identityIdentity:output:0*<
_input_shapes+
):ô::::::2T
(sequential/Output/BiasAdd/ReadVariableOp(sequential/Output/BiasAdd/ReadVariableOp2R
'sequential/Output/MatMul/ReadVariableOp'sequential/Output/MatMul/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:N J
%
_output_shapes
:ô
!
_user_specified_name	input_1
ñ
Î
)__inference_sequential_layer_call_fn_1788

inputs
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
output_kernel
output_bias
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_biasdense_1_kerneldense_1_biasoutput_kerneloutput_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ô*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_16402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	ô2

Identity"
identityIdentity:output:0*<
_input_shapes+
):ô::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
%
_output_shapes
:ô
 
_user_specified_nameinputs
ñ
Î
)__inference_sequential_layer_call_fn_1799

inputs
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
output_kernel
output_bias
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_biasdense_1_kerneldense_1_biasoutput_kerneloutput_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ô*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_16672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	ô2

Identity"
identityIdentity:output:0*<
_input_shapes+
):ô::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
%
_output_shapes
:ô
 
_user_specified_nameinputs
º
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_1887

inputs

identity_1W
IdentityIdentityinputs*
T0*$
_output_shapes
:ô2

Identityf

Identity_1IdentityIdentity:output:0*
T0*$
_output_shapes
:ô2

Identity_1"!

identity_1Identity_1:output:0*#
_input_shapes
:ô:L H
$
_output_shapes
:ô
 
_user_specified_nameinputs

a
(__inference_dropout_1_layer_call_fn_1892

inputs
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_15492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*$
_output_shapes
:ô2

Identity"
identityIdentity:output:0*#
_input_shapes
:ô22
StatefulPartitionedCallStatefulPartitionedCall:L H
$
_output_shapes
:ô
 
_user_specified_nameinputs
Ô	
á
@__inference_Output_layer_call_and_return_conditional_losses_1592

inputs'
#matmul_readvariableop_output_kernel&
"biasadd_readvariableop_output_bias
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOp#matmul_readvariableop_output_kernel*
_output_shapes
:	¨(*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ô2
MatMul
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_output_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ô2	
BiasAddY
SoftmaxSoftmaxBiasAdd:output:0*
T0*
_output_shapes
:	ô2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes
:	ô2

Identity"
identityIdentity:output:0*'
_input_shapes
:
ô¨(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:H D
 
_output_shapes
:
ô¨(
 
_user_specified_nameinputs
ß
£
D__inference_sequential_layer_call_and_return_conditional_losses_1605
input_1
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
output_output_kernel
output_output_bias
identity¢Output/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ôd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_14672
dense/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ôd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_14922!
dropout/StatefulPartitionedCallº
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_15252!
dense_1/StatefulPartitionedCall®
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_15492#
!dropout_1/StatefulPartitionedCallì
flatten/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
ô¨(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_15732
flatten/PartitionedCall¦
Output/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0output_output_kerneloutput_output_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_Output_layer_call_and_return_conditional_losses_15922 
Output/StatefulPartitionedCall
IdentityIdentity'Output/StatefulPartitionedCall:output:0^Output/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*
_output_shapes
:	ô2

Identity"
identityIdentity:output:0*<
_input_shapes+
):ô::::::2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:N J
%
_output_shapes
:ô
!
_user_specified_name	input_1
º
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_1554

inputs

identity_1W
IdentityIdentityinputs*
T0*$
_output_shapes
:ô2

Identityf

Identity_1IdentityIdentity:output:0*
T0*$
_output_shapes
:ô2

Identity_1"!

identity_1Identity_1:output:0*#
_input_shapes
:ô:L H
$
_output_shapes
:ô
 
_user_specified_nameinputs
¸
_
A__inference_dropout_layer_call_and_return_conditional_losses_1838

inputs

identity_1W
IdentityIdentityinputs*
T0*$
_output_shapes
:ôd2

Identityf

Identity_1IdentityIdentity:output:0*
T0*$
_output_shapes
:ôd2

Identity_1"!

identity_1Identity_1:output:0*#
_input_shapes
:ôd:L H
$
_output_shapes
:ôd
 
_user_specified_nameinputs

b
C__inference_dropout_1_layer_call_and_return_conditional_losses_1882

inputs
identityc
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
:ô2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"ô       2
dropout/Shape±
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*$
_output_shapes
:ô*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y»
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*$
_output_shapes
:ô2
dropout/GreaterEqual|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*$
_output_shapes
:ô2
dropout/Castw
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*$
_output_shapes
:ô2
dropout/Mul_1b
IdentityIdentitydropout/Mul_1:z:0*
T0*$
_output_shapes
:ô2

Identity"
identityIdentity:output:0*#
_input_shapes
:ô:L H
$
_output_shapes
:ô
 
_user_specified_nameinputs
Ç

%__inference_Output_layer_call_fn_1926

inputs
output_kernel
output_bias
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinputsoutput_kerneloutput_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_Output_layer_call_and_return_conditional_losses_15922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	ô2

Identity"
identityIdentity:output:0*'
_input_shapes
:
ô¨(::22
StatefulPartitionedCallStatefulPartitionedCall:H D
 
_output_shapes
:
ô¨(
 
_user_specified_nameinputs
À
ê
A__inference_dense_1_layer_call_and_return_conditional_losses_1863

inputs+
'tensordot_readvariableop_dense_1_kernel'
#biasadd_readvariableop_dense_1_bias
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_1_kernel*
_output_shapes

:d*
dtype02
Tensordot/ReadVariableOp
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"è÷ d   2
Tensordot/Reshape/shape
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
èïd2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
èï2
Tensordot/MatMulw
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ô       2
Tensordot/shape
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*$
_output_shapes
:ô2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*$
_output_shapes
:ô2	
BiasAddU
ReluReluBiasAdd:output:0*
T0*$
_output_shapes
:ô2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*$
_output_shapes
:ô2

Identity"
identityIdentity:output:0*+
_input_shapes
:ôd::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:L H
$
_output_shapes
:ôd
 
_user_specified_nameinputs
Ü
¢
D__inference_sequential_layer_call_and_return_conditional_losses_1640

inputs
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
output_output_kernel
output_output_bias
identity¢Output/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ôd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_14672
dense/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ôd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_14922!
dropout/StatefulPartitionedCallº
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_15252!
dense_1/StatefulPartitionedCall®
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_15492#
!dropout_1/StatefulPartitionedCallì
flatten/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
ô¨(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_15732
flatten/PartitionedCall¦
Output/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0output_output_kerneloutput_output_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_Output_layer_call_and_return_conditional_losses_15922 
Output/StatefulPartitionedCall
IdentityIdentity'Output/StatefulPartitionedCall:output:0^Output/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*
_output_shapes
:	ô2

Identity"
identityIdentity:output:0*<
_input_shapes+
):ô::::::2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:M I
%
_output_shapes
:ô
 
_user_specified_nameinputs
m
é
 __inference__traced_restore_2109
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias$
 assignvariableop_4_output_kernel"
assignvariableop_5_output_bias)
%assignvariableop_6_training_adam_iter+
'assignvariableop_7_training_adam_beta_1+
'assignvariableop_8_training_adam_beta_2*
&assignvariableop_9_training_adam_decay3
/assignvariableop_10_training_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count4
0assignvariableop_13_training_adam_dense_kernel_m2
.assignvariableop_14_training_adam_dense_bias_m6
2assignvariableop_15_training_adam_dense_1_kernel_m4
0assignvariableop_16_training_adam_dense_1_bias_m5
1assignvariableop_17_training_adam_output_kernel_m3
/assignvariableop_18_training_adam_output_bias_m4
0assignvariableop_19_training_adam_dense_kernel_v2
.assignvariableop_20_training_adam_dense_bias_v6
2assignvariableop_21_training_adam_dense_1_kernel_v4
0assignvariableop_22_training_adam_dense_1_bias_v5
1assignvariableop_23_training_adam_output_kernel_v3
/assignvariableop_24_training_adam_output_bias_v
identity_26¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¦
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*²
value¨B¥B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÂ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices­
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¢
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¦
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¤
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¥
AssignVariableOp_4AssignVariableOp assignvariableop_4_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5£
AssignVariableOp_5AssignVariableOpassignvariableop_5_output_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6ª
AssignVariableOp_6AssignVariableOp%assignvariableop_6_training_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¬
AssignVariableOp_7AssignVariableOp'assignvariableop_7_training_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¬
AssignVariableOp_8AssignVariableOp'assignvariableop_8_training_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9«
AssignVariableOp_9AssignVariableOp&assignvariableop_9_training_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10·
AssignVariableOp_10AssignVariableOp/assignvariableop_10_training_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¡
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¡
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¸
AssignVariableOp_13AssignVariableOp0assignvariableop_13_training_adam_dense_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¶
AssignVariableOp_14AssignVariableOp.assignvariableop_14_training_adam_dense_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15º
AssignVariableOp_15AssignVariableOp2assignvariableop_15_training_adam_dense_1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¸
AssignVariableOp_16AssignVariableOp0assignvariableop_16_training_adam_dense_1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¹
AssignVariableOp_17AssignVariableOp1assignvariableop_17_training_adam_output_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18·
AssignVariableOp_18AssignVariableOp/assignvariableop_18_training_adam_output_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¸
AssignVariableOp_19AssignVariableOp0assignvariableop_19_training_adam_dense_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¶
AssignVariableOp_20AssignVariableOp.assignvariableop_20_training_adam_dense_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21º
AssignVariableOp_21AssignVariableOp2assignvariableop_21_training_adam_dense_1_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¸
AssignVariableOp_22AssignVariableOp0assignvariableop_22_training_adam_dense_1_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¹
AssignVariableOp_23AssignVariableOp1assignvariableop_23_training_adam_output_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24·
AssignVariableOp_24AssignVariableOp/assignvariableop_24_training_adam_output_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_249
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25÷
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_26"#
identity_26Identity_26:output:0*y
_input_shapesh
f: :::::::::::::::::::::::::2$
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
å
Ý
D__inference_sequential_layer_call_and_return_conditional_losses_1621
input_1
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
output_output_kernel
output_output_bias
identity¢Output/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ôd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_14672
dense/StatefulPartitionedCallì
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ôd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_14972
dropout/PartitionedCall²
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_15252!
dense_1/StatefulPartitionedCallô
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:ô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_15542
dropout_1/PartitionedCallä
flatten/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
ô¨(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_15732
flatten/PartitionedCall¦
Output/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0output_output_kerneloutput_output_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_Output_layer_call_and_return_conditional_losses_15922 
Output/StatefulPartitionedCallÖ
IdentityIdentity'Output/StatefulPartitionedCall:output:0^Output/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*
_output_shapes
:	ô2

Identity"
identityIdentity:output:0*<
_input_shapes+
):ô::::::2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:N J
%
_output_shapes
:ô
!
_user_specified_name	input_1

]
A__inference_flatten_layer_call_and_return_conditional_losses_1903

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(  2
Const`
ReshapeReshapeinputsConst:output:0*
T0* 
_output_shapes
:
ô¨(2	
Reshape]
IdentityIdentityReshape:output:0*
T0* 
_output_shapes
:
ô¨(2

Identity"
identityIdentity:output:0*#
_input_shapes
:ô:L H
$
_output_shapes
:ô
 
_user_specified_nameinputs
Ô	
á
@__inference_Output_layer_call_and_return_conditional_losses_1919

inputs'
#matmul_readvariableop_output_kernel&
"biasadd_readvariableop_output_bias
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOp#matmul_readvariableop_output_kernel*
_output_shapes
:	¨(*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ô2
MatMul
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_output_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ô2	
BiasAddY
SoftmaxSoftmaxBiasAdd:output:0*
T0*
_output_shapes
:	ô2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes
:	ô2

Identity"
identityIdentity:output:0*'
_input_shapes
:
ô¨(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:H D
 
_output_shapes
:
ô¨(
 
_user_specified_nameinputs
ø
B
&__inference_flatten_layer_call_fn_1908

inputs
identity¸
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
ô¨(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_15732
PartitionedCalle
IdentityIdentityPartitionedCall:output:0*
T0* 
_output_shapes
:
ô¨(2

Identity"
identityIdentity:output:0*#
_input_shapes
:ô:L H
$
_output_shapes
:ô
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
9
input_1.
serving_default_input_1:0ô2
Output(
StatefulPartitionedCall:0	ôtensorflow/serving/predict:ç·
­(
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
	optimizer
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
e_default_save_signature
*f&call_and_return_all_conditional_losses
g__call__"Æ%
_tf_keras_sequential§%{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [500, 258, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "Output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [500, 258, 128]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [500, 258, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "Output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "loss_weights": null, "sample_weight_mode": null, "weighted_metrics": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ó

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*h&call_and_return_all_conditional_losses
i__call__"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [500, 258, 128]}}
á
trainable_variables
regularization_losses
	variables
	keras_api
*j&call_and_return_all_conditional_losses
k__call__"Ò
_tf_keras_layer¸{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
ö

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*l&call_and_return_all_conditional_losses
m__call__"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [500, 258, 100]}}
å
trainable_variables
regularization_losses
	variables
 	keras_api
*n&call_and_return_all_conditional_losses
o__call__"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
â
!trainable_variables
"regularization_losses
#	variables
$	keras_api
*p&call_and_return_all_conditional_losses
q__call__"Ó
_tf_keras_layer¹{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ó

%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
*r&call_and_return_all_conditional_losses
s__call__"Î
_tf_keras_layer´{"class_name": "Dense", "name": "Output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5160}}}, "build_input_shape": {"class_name": "TensorShape", "items": [500, 5160]}}
¿
+iter

,beta_1

-beta_2
	.decay
/learning_ratemYmZm[m\%m]&m^v_v`vavb%vc&vd"
	optimizer
J
0
1
2
3
%4
&5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
%4
&5"
trackable_list_wrapper
Ê

0layers
trainable_variables
1layer_regularization_losses
2non_trainable_variables
	regularization_losses

	variables
3layer_metrics
4metrics
g__call__
e_default_save_signature
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
,
tserving_default"
signature_map
:	d2dense/kernel
:d2
dense/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

5layers
trainable_variables
6layer_regularization_losses
7non_trainable_variables
regularization_losses
	variables
8layer_metrics
9metrics
i__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

:layers
trainable_variables
;layer_regularization_losses
<non_trainable_variables
regularization_losses
	variables
=layer_metrics
>metrics
k__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
 :d2dense_1/kernel
:2dense_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

?layers
trainable_variables
@layer_regularization_losses
Anon_trainable_variables
regularization_losses
	variables
Blayer_metrics
Cmetrics
m__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

Dlayers
trainable_variables
Elayer_regularization_losses
Fnon_trainable_variables
regularization_losses
	variables
Glayer_metrics
Hmetrics
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

Ilayers
!trainable_variables
Jlayer_regularization_losses
Knon_trainable_variables
"regularization_losses
#	variables
Llayer_metrics
Mmetrics
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
 :	¨(2Output/kernel
:2Output/bias
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
­

Nlayers
'trainable_variables
Olayer_regularization_losses
Pnon_trainable_variables
(regularization_losses
)	variables
Qlayer_metrics
Rmetrics
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
:	 (2training/Adam/iter
: (2training/Adam/beta_1
: (2training/Adam/beta_2
: (2training/Adam/decay
%:# (2training/Adam/learning_rate
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
S0"
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
ÿ
	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api"¸
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
T0
U1"
trackable_list_wrapper
-
W	variables"
_generic_user_object
-:+	d2training/Adam/dense/kernel/m
&:$d2training/Adam/dense/bias/m
.:,d2training/Adam/dense_1/kernel/m
(:&2training/Adam/dense_1/bias/m
.:,	¨(2training/Adam/Output/kernel/m
':%2training/Adam/Output/bias/m
-:+	d2training/Adam/dense/kernel/v
&:$d2training/Adam/dense/bias/v
.:,d2training/Adam/dense_1/kernel/v
(:&2training/Adam/dense_1/bias/v
.:,	¨(2training/Adam/Output/kernel/v
':%2training/Adam/Output/bias/v
Û2Ø
__inference__wrapped_model_1448´
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *$¢!

input_1ô
Þ2Û
D__inference_sequential_layer_call_and_return_conditional_losses_1777
D__inference_sequential_layer_call_and_return_conditional_losses_1740
D__inference_sequential_layer_call_and_return_conditional_losses_1621
D__inference_sequential_layer_call_and_return_conditional_losses_1605À
·²³
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
)__inference_sequential_layer_call_fn_1676
)__inference_sequential_layer_call_fn_1649
)__inference_sequential_layer_call_fn_1788
)__inference_sequential_layer_call_fn_1799À
·²³
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
é2æ
?__inference_dense_layer_call_and_return_conditional_losses_1814¢
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
Î2Ë
$__inference_dense_layer_call_fn_1821¢
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
À2½
A__inference_dropout_layer_call_and_return_conditional_losses_1838
A__inference_dropout_layer_call_and_return_conditional_losses_1833´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
&__inference_dropout_layer_call_fn_1848
&__inference_dropout_layer_call_fn_1843´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ë2è
A__inference_dense_1_layer_call_and_return_conditional_losses_1863¢
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
Ð2Í
&__inference_dense_1_layer_call_fn_1870¢
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
Ä2Á
C__inference_dropout_1_layer_call_and_return_conditional_losses_1887
C__inference_dropout_1_layer_call_and_return_conditional_losses_1882´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
(__inference_dropout_1_layer_call_fn_1897
(__inference_dropout_1_layer_call_fn_1892´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ë2è
A__inference_flatten_layer_call_and_return_conditional_losses_1903¢
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
Ð2Í
&__inference_flatten_layer_call_fn_1908¢
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
ê2ç
@__inference_Output_layer_call_and_return_conditional_losses_1919¢
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
Ï2Ì
%__inference_Output_layer_call_fn_1926¢
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
ÉBÆ
"__inference_signature_wrapper_1689input_1"
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
 
@__inference_Output_layer_call_and_return_conditional_losses_1919M%&(¢%
¢

inputs
ô¨(
ª "¢

0	ô
 i
%__inference_Output_layer_call_fn_1926@%&(¢%
¢

inputs
ô¨(
ª "	ô
__inference__wrapped_model_1448a%&.¢+
$¢!

input_1ô
ª "'ª$
"
Output
Output	ô
A__inference_dense_1_layer_call_and_return_conditional_losses_1863V,¢)
"¢

inputsôd
ª ""¢

0ô
 s
&__inference_dense_1_layer_call_fn_1870I,¢)
"¢

inputsôd
ª "ô
?__inference_dense_layer_call_and_return_conditional_losses_1814W-¢*
#¢ 

inputsô
ª ""¢

0ôd
 r
$__inference_dense_layer_call_fn_1821J-¢*
#¢ 

inputsô
ª "ôd
C__inference_dropout_1_layer_call_and_return_conditional_losses_1882V0¢-
&¢#

inputsô
p
ª ""¢

0ô
 
C__inference_dropout_1_layer_call_and_return_conditional_losses_1887V0¢-
&¢#

inputsô
p 
ª ""¢

0ô
 u
(__inference_dropout_1_layer_call_fn_1892I0¢-
&¢#

inputsô
p
ª "ôu
(__inference_dropout_1_layer_call_fn_1897I0¢-
&¢#

inputsô
p 
ª "ô
A__inference_dropout_layer_call_and_return_conditional_losses_1833V0¢-
&¢#

inputsôd
p
ª ""¢

0ôd
 
A__inference_dropout_layer_call_and_return_conditional_losses_1838V0¢-
&¢#

inputsôd
p 
ª ""¢

0ôd
 s
&__inference_dropout_layer_call_fn_1843I0¢-
&¢#

inputsôd
p
ª "ôds
&__inference_dropout_layer_call_fn_1848I0¢-
&¢#

inputsôd
p 
ª "ôd
A__inference_flatten_layer_call_and_return_conditional_losses_1903N,¢)
"¢

inputsô
ª "¢

0
ô¨(
 k
&__inference_flatten_layer_call_fn_1908A,¢)
"¢

inputsô
ª "
ô¨(§
D__inference_sequential_layer_call_and_return_conditional_losses_1605_%&6¢3
,¢)

input_1ô
p

 
ª "¢

0	ô
 §
D__inference_sequential_layer_call_and_return_conditional_losses_1621_%&6¢3
,¢)

input_1ô
p 

 
ª "¢

0	ô
 ¦
D__inference_sequential_layer_call_and_return_conditional_losses_1740^%&5¢2
+¢(

inputsô
p

 
ª "¢

0	ô
 ¦
D__inference_sequential_layer_call_and_return_conditional_losses_1777^%&5¢2
+¢(

inputsô
p 

 
ª "¢

0	ô
 
)__inference_sequential_layer_call_fn_1649R%&6¢3
,¢)

input_1ô
p

 
ª "	ô
)__inference_sequential_layer_call_fn_1676R%&6¢3
,¢)

input_1ô
p 

 
ª "	ô~
)__inference_sequential_layer_call_fn_1788Q%&5¢2
+¢(

inputsô
p

 
ª "	ô~
)__inference_sequential_layer_call_fn_1799Q%&5¢2
+¢(

inputsô
p 

 
ª "	ô
"__inference_signature_wrapper_1689l%&9¢6
¢ 
/ª,
*
input_1
input_1ô"'ª$
"
Output
Output	ô