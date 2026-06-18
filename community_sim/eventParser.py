def ParseControlEvent(controlEvent):
    '''
    Parses control events in the format of the controls API for deployment
    Controls API format: [{location: house1, devices: {id:on, id:off}}, {location:house2}, etc]
    Simulation formation: [{controlType: battery, houses: {1,2,3}}, {controlType: HVAC}, etc]
    Inputs:
        controlEvents (list[dict]) list of houses and the devices with control events
    Outputs:
        controlEvents ()
    '''
    pass