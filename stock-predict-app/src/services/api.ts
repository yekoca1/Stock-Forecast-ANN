export interface StockHistoryItem {
  date: string
  close: number
}

export interface StockPredictions {
  ticker: any
  tomorrow: number
  oneMonth: number
}

export interface FundamentalMetric {
  value: number | null
  rating: string
}

export interface Fundamentals {
  forwardPE: FundamentalMetric
  priceToBook: FundamentalMetric
  returnOnEquity: FundamentalMetric
  debtToEquity: FundamentalMetric
}

export interface StockApiResponse {
  history: StockHistoryItem[]
  predictions: StockPredictions
  fundamentals: Fundamentals
  rsi: number
}

export async function fetchStockData(stockName: string): Promise<StockApiResponse> {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ ticker: stockName }),
  })

  if (!response.ok) {
    const errorData = await response.json()
    throw new Error(errorData.detail || 'Sunucu hatası oluştu.')
  }

  const data: StockApiResponse = await response.json()
  return data
}
