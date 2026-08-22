from hublink.core.models.hub_link_settings import (
    ADDITIONAL_CONFIG_PARAMS,
    HubLinkSettings,
)


def test_cross_encoder_is_disabled_by_default():
    assert HubLinkSettings().use_cross_encoder is False


def test_cross_encoder_can_be_enabled_from_additional_config():
    parameter = next(
        param
        for param in ADDITIONAL_CONFIG_PARAMS
        if param.name == "use_cross_encoder"
    )

    assert parameter.parse_value("true") is True
    assert HubLinkSettings(use_cross_encoder=True).use_cross_encoder is True
