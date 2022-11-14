

REWARD_BY_LIST=['reward','mdd','1mstdev']#reward_by_list
TOLERANCE_RANGE = 0.1 #tolerance_range

GET_DATA_TRAIN_START = '2008-01-01' # get_data_train_start = org_train_start - 1year
ORG_TRAIN_START = '2009-01-01' #org_train_start
ORG_TRAIN_END = '2016-01-01' #org_train_end
GET_DATA_TRADE_START = '2015-01-01' # get_data_trade_start = org_trade_start - 1year
ORG_TRADE_START = '2016-01-01' #org_trade_start
ORG_TRADE_END = '2021-12-31' #org_trade_end
ORG_INIT_AMOUNT = 1000000#org_initial_amount
COLORS = ['b','orange','g','r','purple','brown','pink','gray','olive','cyan','m','y','black','lime','greenyellow','darksalmon']


COMBS = {
    'David Swensen':{ 
        'etfs':['VTI','VEA','EEM','TLT','TIP','VNQ'],
        'weights':[0.2,0.2,0.1,0.15,0.15,0.2]
    },
    'All Weather':{ 
        'etfs':['VTI','TLT','IEF','GLD','DBC'],
        'weights':[0.3,0.4,0.15,0.075,0.075]
    },
    'Golden Butterfly':{ 
        'etfs':['IJS','VTI','TLT','SHY','GLD'],
        'weights':[0.2,0.2,0.2,0.2,0.2]
    },
    'Paul Merriman':{ 
        'etfs':['IJS','DLS','EEM','SPY','IJR','VTV','VEA','SCZ','EFV','VNQ'],
        'weights':[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    },
    'highhigh':{ 
        'etfs':['XBI','SAA','IWC','IWV'],
        'weights':[0.25,0.25,0.25,0.25]
    },
    'good':{ 
        'etfs':['PRF','FTC','IUSV','DLN'],
        'weights':[0.25,0.25,0.25,0.25]
    },
    'lowlow':{ 
        'etfs':['IYG','IWM','IWD','IXC'],
        'weights':[0.25,0.25,0.25,0.25]
    },
    'bad':{ 
        'etfs':['UWM','IWV','GWX','IWC'],
        'weights':[0.25,0.25,0.25,0.25]
    }
}


