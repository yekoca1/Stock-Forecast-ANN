export interface StockHistoryItem {
    date: string;
    close: number;
  }
  
  export interface StockPrediction {
    tomorrow: number;
    oneWeek: number;
    oneMonth: number;
  }
  
