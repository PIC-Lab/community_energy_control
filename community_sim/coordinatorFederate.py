import logging
import helics as h
import os

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

fed = h.helicsCreateCombinationFederateFromConfig(
    os.path.join(os.path.dirname(__file__), "coordinatorFederate.json")
)
federate_name = h.helicsFederateGetName(fed)
logger.info(f"Created federate {federate_name}")

sub_count = h.helicsFederateGetInputCount(fed)
logger.debug(f"\tNumber of subscriptions: {sub_count}")
pub_count = h.helicsFederateGetPublicationCount(fed)
logger.debug(f"\tNumber of publications: {pub_count}")