require 'alfalfa'
require 'json'

class ExampleModelMeasure < OpenStudio::Measure::ModelMeasure

  include OpenStudio::Alfalfa::OpenStudioMixin
  
  # human readable name
  def name
    return 'Example Model Measure'
  end

  # human readable description
  def description
    return 'An Example Measure to act as a starting point for a tutorial'
  end

  # human readable description of modeling approach
  def modeler_description
    return 'An Example Measure to act as a starting point for a tutorial'
  end

  # define the arguments that the user will input
  def arguments(model)
    OpenStudio::Measure::OSArgumentVector.new
  end

  # define what happens when the measure is run
  def run(model, runner, user_arguments)
    super(model, runner, user_arguments)  # Do **NOT** remove this line

    # use the built-in error checking
    if !runner.validateUserArguments(arguments(model), user_arguments)
      return false
    end

    # Change temperature capacity multiplier
    zoneCap = model.getZoneCapacitanceMultiplierResearchSpecial
    zoneCap.setTemperatureCapacityMultiplier(3)

    # Set thermostat deadband
    thermostats = model.getThermostats
    thermostats.each do |thermostat|
      dualSetpoint = thermostat.thermalZone.get.thermostatSetpointDualSetpoint.get
      dualSetpoint.setTemperatureDifferenceBetweenCutoutAndSetpoint(1.0)
    end

    # Water heater
    waterHeaters = model.getWaterHeaterHeatPumpWrappedCondensers
    waterHeaters.each do |waterHeater|
      name = waterHeater.name.get

      # Create actuator for water heater setpoint
      waterHeaterSch = waterHeater.compressorSetpointTemperatureSchedule
      waterHeater_actuator = create_actuator(create_ems_str('Water Heater Setpoint'), waterHeaterSch, 
                                           'Schedule:Constant', 'Schedule Value', true)
      waterHeater_actuator.display_name = "#{name} Setpoint"
      register_input(waterHeater_actuator)

      # waterHeater_setpoint = create_output_variable(waterHeaterSch.name.get, 'Schedule Value')
      # waterHeater_actuator.echo = waterHeater_setpoint

      # Create output for water heater tank temperature
      # hpwhTemp = create_output_variable('*', 'Water Heater Tank Temperature')
      # hpwhTemp.display_name = "#{name} Tank Temperature"
      # register_output(hpwhTemp)
    end

    # Add output for outdoor air temperature
    name = model.outdoorAirNode.name.get
    outdoorTemp = create_output_variable(name, 'System Node Temperature')
    outdoorTemp.display_name = "Site Outdoor Air Temperature"
    register_output(outdoorTemp)

    # Heating and cooling rates
    # heatingCoils = model.getCoilCoolingDXMultiSpeeds
    # heatingCoils.each do |heatingCoil|
    #   name = heatingCoil.name.get
    #   heatingRate = create_output_variable(name, 'Heating Coil Heating Rate')
    #   heatingRate.display_name = '#{name} Heating Rate'
    #   register_output(heatingRate)
    # end

    # coolingCoils = model.getCoilHeatingDXMultiSpeeds
    # coolingCoils.each do |coolingCoil|
    #   name = coolingCoil.name.get
    #   coolingRate = create_output_variable(name, 'Cooling Coil Sensible Cooling Rate')
    #   coolingRate.display_name = '#{name} Sensible Cooling Rate'
    #   register_output(coolingRate)
    # end
    
    # name = model.getFacility
    # facilityElectricity = create_output_variable(name, 'Electricity')
    # facilityElectricity.display_name = "Facility Electricity"
    # register_output(facilityElectricity)
    
    # Has to be included for the inputs and outputs to show up in Alfalfa
    report_inputs_outputs

    # Not sure if this is needed
    runner.registerFinalCondition("Done")

    return true
  end
end

# register the measure to be used by the application
ExampleModelMeasure.new.registerWithApplication
