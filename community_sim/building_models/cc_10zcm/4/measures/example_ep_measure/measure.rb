require 'alfalfa'

class ExampleEPMeasure < OpenStudio::Measure::EnergyPlusMeasure
  
  include OpenStudio::Alfalfa::EnergyPlusMixin
  
  def name
    "Example EnergyPlus Measure"
  end

  def description
    "An Example Measure to act as a starting point for a tutorial"
  end

  def modeler_description
    "An Example Measure to act as a starting point for a tutorial"
  end

  def arguments(model)
    OpenStudio::Measure::OSArgumentVector.new
  end

  def create_output_meter(name)
    sensor_name = "#{create_ems_str(name)}_sensor"
    output_name = "#{create_ems_str(name)}_output"
    new_meter_string = "
    Output:Meter,
      #{name};
    "
    new_meter_object = OpenStudio::IdfObject.load(new_meter_string).get
    @workspace.addObject(new_meter_object)

    new_sensor_string = "
    EnergyManagementSystem:Sensor,
      #{sensor_name},
      ,
      #{name};
    "
    new_sensor_object = OpenStudio::IdfObject.load(new_sensor_string).get
    @workspace.addObject(new_sensor_object)

    create_ems_output_variable(output_name, sensor_name)
  end

  def run(workspace, runner, user_arguments)
    super(workspace, runner, user_arguments)

    # use the built-in error checking
    if !runner.validateUserArguments(arguments(workspace), user_arguments)
      return false
    end

    endUses = [
      'Heating', 'WaterSystems', 'Fans', 'InteriorEquipment'
    ]

    endUses.each do |endUse|
      register_output(create_output_meter("#{endUse}:Electricity")).display_name = "#{endUse}:Electricity"
    end

    register_output(create_output_meter("Electricity:HVAC")).display_name = "Electricity:HVAC"

    # Heating and cooling coils
    # heating_coils = workspace.getObjectsByType('Coil:Heating:DX:MultiSpeed'.to_IddObjectType)
    # heating_coils.each do |heating_coil|
    #   name = heating_coil.name.get
    #   heating_rate = create_output_variable(name, 'Heating Coil Heating Rate')
    #   heating_rate.display_name = "#{name} Heating Rate"
    #   register_output(heating_rate)
    # end

    # cooling_coils = workspace.getObjectsByType('Coil:Cooling:DX:MultiSpeed'.to_IddObjectType)
    # cooling_coils.each do |cooling_coil|
    #   name = cooling_coil.name.get
    #   cooling_rate = create_output_variable(name, 'Cooling Coil Sensible Cooling Rate')
    #   cooling_rate.display_name = "#{name} Cooling Rate"
    #   register_output(cooling_rate)
    # end

    report_inputs_outputs

    runner.registerFinalCondition("Done")

    return true
  end
end

ExampleEPMeasure.new.registerWithApplication
