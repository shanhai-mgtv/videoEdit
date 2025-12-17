from models.transformer_wan import WanTransformer3DModel, WanTransformerBlock


def get_no_split_modules(transformer):
    # if of type WanTransformer3DModel
    if isinstance(transformer, WanTransformer3DModel):
        return (WanTransformerBlock, )
    else:
        raise ValueError(f"Unsupported transformer type: {type(transformer)}")
