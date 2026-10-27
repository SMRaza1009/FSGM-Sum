from models.FSGM import FSGMSum


# Build FSGM-Sum from a Config object
def build_model(config):
    if config.model_name not in ['FSGM', 'FSGM_Attention']:
        raise ValueError(f"Unknown model_name={config.model_name}")

    return FSGMSum(
        model_name=config.model_name,
        Scale=config.Scale,
        Softmax_axis=config.Softmax_axis,
        Balance=config.Balance,
        Positional_encoding=config.Positional_encoding,
        Positional_encoding_shape=config.Positional_encoding_shape,
        Positional_encoding_way=config.Positional_encoding_way,
        Dropout_on=config.Dropout_on,
        Dropout_ratio=config.Dropout_ratio,
        Classifier_on=config.Classifier_on,
        CLS_on=config.CLS_on,
        CLS_mix=config.CLS_mix,
        key_value_emb=config.key_value_emb,
        Skip_connection=config.Skip_connection,
        Layernorm=config.Layernorm,
        fsgm_hidden=config.fsgm_hidden,
        fsgm_dilations=tuple(config.fsgm_dilations),
        fsgm_fft_bias=config.fsgm_fft_bias,
        fsgm_branches=config.fsgm_branches,
        fsgm_pooling=config.fsgm_pooling,
        fsgm_fusion=config.fsgm_fusion,
        temp_learnable=config.temp_learnable,
    )
