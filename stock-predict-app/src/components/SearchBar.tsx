'use client'
import { useState } from 'react';

interface Props {
  onSearch: (stockName: string) => void;
}

function SearchBar({ onSearch }: Props) {
  const [stockName, setStockName] = useState('');

  return (
    <div className="flex items-center justify-center gap-2 my-6">
      <input
        value={stockName}
        onChange={(e) => setStockName(e.target.value)}
        placeholder="Hisse Adı Girin"
        className="border p-2 rounded-md"
      />
      <button
        onClick={() => onSearch(stockName)}
        className="bg-blue-500 text-white p-2 rounded-md hover:bg-blue-600"
      >
        Ara
      </button>
    </div>
  );
}


export default SearchBar