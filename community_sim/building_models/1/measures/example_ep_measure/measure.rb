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

    report_inputs_outputs

    runner.registerFinalCondition("Done")

    return true
  end
end

ExampleEPMeasure.new.registerWithApplication