import subprocess
import slackNotifier

if __name__ == '__main__':
    notifier = slackNotifier.SlackNotifier('Leadville Community Sim', ['U04J6NQG084'])
    try:
        
        notifier.Start()
        result = subprocess.call(['sh', './helicsRunner.sh'])
        print(result.returncode)
        print(result.stdout)
        notifier.Stop()
    except:
        notifier.Error()