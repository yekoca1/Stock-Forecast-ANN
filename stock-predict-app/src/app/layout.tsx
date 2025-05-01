import './globals.css'
import { Inter } from 'next/font/google'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Stock Predict App',
  description: 'AI Based Stock Predictor',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="tr">
      <body className={inter.className}>
        <header className="text-center py-4 font-bold text-xl">
          📈 Stock Predict App
        </header>
        <main className="px-4">{children}</main>
      </body>
    </html>
  )
}

