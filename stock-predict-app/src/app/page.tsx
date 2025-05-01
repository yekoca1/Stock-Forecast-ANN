'use client'

import { useState } from 'react'
import SearchBar from '@/components/SearchBar'
import StockChart from '@/components/StockChart'
import Predictions from '@/components/Predictions'
import { fetchStockData } from '@/services/api'
import {
  StockHistoryItem,
  StockPredictions,
  Fundamentals,
} from '@/services/api'

export default function HomePage() {
  const [stockHistory, setStockHistory] = useState<StockHistoryItem[] | null>(null)
  const [predictions, setPredictions] = useState<StockPredictions | null>(null)
  const [fundamentals, setFundamentals] = useState<Fundamentals | null>(null)
  const [rsi, setRsi] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState<boolean>(false)

  const handleSearch = async (stockName: string) => {
    if (!stockName.trim()) {
      setError('Hisse adı boş olamaz.')
      return
    }

    try {
      setLoading(true)
      setError(null)

      const data = await fetchStockData(stockName)

      if (!data || !data.history?.length || !data.predictions) {
        throw new Error('Hisse bulunamadı veya veri eksik.')
      }

      setStockHistory(data.history)
      setPredictions(data.predictions)
      setFundamentals(data.fundamentals)
      setRsi(data.rsi)
    } catch (err: any) {
      console.error(err)
      setError(err.message || 'Bir hata oluştu.')
      setStockHistory(null)
      setPredictions(null)
      setFundamentals(null)
      setRsi(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col items-center justify-center mt-8">
      {/* 🔎 Search */}
      <SearchBar onSearch={handleSearch} />

      {/* 📊 Loading state */}
      {loading && <div className="text-blue-500 mt-4">Yükleniyor...</div>}

      {/* ❗ Error message */}
      {error && <div className="text-red-500 mt-4">{error}</div>}

      {/* 📈 Stock Chart */}
      {stockHistory && !loading && (
        <div className="w-full max-w-4xl mt-8">
          <h2 className="text-lg font-semibold mb-2">📚 Hisse Geçmişi (Grafik)</h2>
          <StockChart data={stockHistory} />
        </div>
      )}

      {/* 📅 Predictions */}
      {predictions && !loading && (
        <div className="w-full max-w-md mt-8">
          <Predictions
            predictions={predictions}
            // Optional: Add props for rsi & fundamentals if Predictions will use them
          />
        </div>
      )}
    </div>
  )
}
