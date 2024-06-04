import dill
from apscheduler.schedulers.blocking import BlockingScheduler
import tzlocal
import pandas as pd


sched = BlockingScheduler(timezone=tzlocal.get_localzone())

df = pd.read_csv('model/data/homework.csv')
with open('model/cars_pipe.pkl', 'rb') as file:
    model = dill.load(file)


@sched.scheduled_job('cron', second='*/5')
def on_tim():
    data = df.sample(frac=0.0005)
    data['predicted_price_cat'] = model['model'].predict(data)
    print(data[['id', 'price', 'predicted_price_cat']])


if __name__ == '__main__':
    sched.start()